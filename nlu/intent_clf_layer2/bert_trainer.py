import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from loguru import logger
from .models import MyBert
from .utils import BASE_MODEL_PATH,MODEL_PATH,LOG_PATH,TRAIN_PATH, VAL_PATH


class IntentDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        self.text = data["text"].tolist()
        self.labels = data["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        x = self.text[item]
        y = self.labels[item]

        return (x, y)


class IntentTrainer(object):
    def __init__(self, pretrained_model_dir=None, model_save_dir=None, ckpt_name=None, learning_rate=1e-5):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_save_dir = model_save_dir
        self.learning_rate = learning_rate
        self.ckpt_name = ckpt_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = None
        self.epoch = None

        self.model = MyBert(self.pretrained_model_dir)

        # 设置随机种子
        seed = 2022
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def collate_fn(self, train_data):
        train_texts = [data[0] for data in train_data]
        train_labels = [data[1] for data in train_data]

        max_len = max([len(data[0]) for data in train_data]) + 2  # cls+sep
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        for text, label in zip(train_texts, train_labels):
            assert isinstance(text, str)
            bert_tokenizer = self.model.tokenizer()
            encoded_dict = bert_tokenizer.encode_plus(text, max_length=max_len, padding="max_length",
                                                     return_tensors="pt", truncation=True)
            batch_input_ids.append(encoded_dict["input_ids"])
            batch_attention_mask.append(encoded_dict["attention_mask"])
            batch_labels.append(label)

        batch_input_ids = torch.cat(batch_input_ids)
        batch_attention_mask = torch.cat(batch_attention_mask)
        batch_labels = torch.LongTensor(batch_labels)

        return batch_input_ids, batch_attention_mask, batch_labels

    def train(self, train_data_path, val_data_path, batch_size, epoch, early_stop=50):
        self.batch_size = batch_size
        self.epoch = epoch
        self.early_stop = early_stop
        early_num = 0
        step = 0
        stop_tag = False
        train_loss_record = []
        val_loss_record = []

        best_loss = float("inf")

        train_dataset = IntentDataset(train_data_path)
        val_dataset = IntentDataset(val_data_path)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        t_total = len(train_dataloader) * epoch
        no_decay = ["bias", "LayerNorm.weight"]  # 对bias和LayerNorm不使用正则化
        # 区分bert层参数和全连接层参数
        bert_parameters = [(name, param) for name, param in self.model.named_parameters() if "bert" in name]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=t_total,
            num_warmup_steps=100,
            power=2
        )
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_training_steps=t_total,
        #     num_warmup_steps=50,
        #     num_cycles=t_total//1000
        # )
        self.model.to(self.device)
        self.model.zero_grad()

        scaler = GradScaler()

        for _epoch in range(epoch):
            for idx, batch_train in enumerate(train_dataloader):
                step += 1
                self.model.train()
                # self.model.zero_grad()

                batch_input_ids, batch_attention_mask, batch_labels = batch_train[0].to(self.device), batch_train[1].to(self.device), batch_train[2].to(self.device)

                # simCSE，复制一份样本，确保对每个样本而言batch中有至少一个同类样本 -> 确保infoNCE分子不为0
                batch_input_ids = torch.cat([batch_input_ids,batch_input_ids])
                batch_attention_mask = torch.cat([batch_attention_mask,batch_attention_mask])
                batch_labels = torch.cat([batch_labels,batch_labels])

                with autocast():
                    loss = self.model(batch_input_ids, batch_attention_mask, labels=batch_labels)
                    # 梯度累计
                    loss /= 2
                scaler.scale(loss).backward()
                # 经过梯度裁剪会损失精度
                # clip_grad_norm_(self.model.parameters(), 5)

                if (idx+1) % 2 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                if step % 10 == 0:
                    self.model.eval()
                    batch_val_loss = []

                    for batch_val in val_dataloader:
                        with torch.no_grad():
                            batch_input_ids_val, batch_attention_mask_val, batch_labels_val = batch_val[0].to(self.device), batch_val[1].to(self.device), batch_val[2].to(self.device)
                            
                            # simCSE，复制一份样本，确保对每个样本而言batch中有至少一个同类样本 -> 确保infoNCE分子不为0
                            batch_input_ids_val = torch.cat([batch_input_ids_val,batch_input_ids_val])
                            batch_attention_mask_val = torch.cat([batch_attention_mask_val,batch_attention_mask_val])
                            batch_labels_val = torch.cat([batch_labels_val,batch_labels_val])

                            loss_val = self.model(batch_input_ids_val, batch_attention_mask_val, labels=batch_labels_val)
                            batch_val_loss.append(loss_val)

                    avg_loss = sum(batch_val_loss) / len(batch_val_loss)
                    train_loss_record.append(loss*2)
                    val_loss_record.append(avg_loss)

                    logger.info("\nepoch %d, step %d, train_loss %.4f, val_loss %.4f" % (_epoch, step, loss*2, avg_loss))
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        early_num = 0
                        state_dict = self.model.state_dict()
                        torch.save(state_dict, "{}/{}".format(self.model_save_dir, self.ckpt_name))
                        logger.info("\nVal loss decreased, model saved as %s/%s. Epoch %d, Step %d, Train_loss %.4f, Val_loss %.4f" % (self.model_save_dir,self.ckpt_name,_epoch, step, loss*2, avg_loss))
                    else:
                        early_num += 1
                        if early_num == self.early_stop:
                            stop_tag = True
                            break
            if stop_tag:
                logger.info("\nEarly Stopping. Epoch %d, Step %d" % (_epoch, step))
                break
        
        return (train_loss_record, val_loss_record)


if __name__ == '__main__':
    ckpt_name = "retrained_bert.bin"
    trainer = IntentTrainer(pretrained_model_dir=BASE_MODEL_PATH, model_save_dir=MODEL_PATH, ckpt_name=ckpt_name)
    batch_size = 16
    epoch = 25
    logger.add(LOG_PATH+"/retrained_bert.log", rotation='10MB')
    logger.info("\nStart training with batch_size=%d, epoch=%d..." % (batch_size, epoch))
    trainer.train(TRAIN_PATH, VAL_PATH, batch_size=batch_size, epoch=epoch, early_stop=100)
    logger.info("\nTraining finished...")
    logger.remove()