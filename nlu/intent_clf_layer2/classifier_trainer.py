import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from loguru import logger
from sklearn.metrics import accuracy_score
from .models import MyBert, Classifier
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
    def __init__(self, pretrained_model_dir=None, model_save_dir=None, encoder_ckpt_name=None, decoder_ckpt_name=None, learning_rate=1e-2, use_cnn=True):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_save_dir = model_save_dir
        self.learning_rate = learning_rate
        self.encoder_ckpt_name = encoder_ckpt_name
        self.decoder_ckpt_name = decoder_ckpt_name
        self.use_cnn = use_cnn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = None
        self.epoch = None

        self.encoder_dir = self.model_save_dir + '/' + self.encoder_ckpt_name
        self.encoder = MyBert(self.pretrained_model_dir)
        self.encoder.load_state_dict(torch.load(self.encoder_dir, map_location=self.device))
        self.decoder = Classifier(use_cnn=self.use_cnn)

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
            bert_tokenizer = self.encoder.tokenizer()
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
        # val_loss_record = []
        val_acc_record = []

        # best_loss = float("inf")
        best_acc = float(0)

        train_dataset = IntentDataset(train_data_path)
        val_dataset = IntentDataset(val_data_path)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        t_total = len(train_dataloader) * epoch
        optimizer = optim.AdamW(self.decoder.parameters())
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
        self.encoder.to(self.device)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.decoder.to(self.device)
        self.decoder.zero_grad()

        scaler = GradScaler()

        for _epoch in range(epoch):
            for idx, batch_train in enumerate(train_dataloader):
                step += 1
                self.decoder.train()

                batch_input_ids, batch_attention_mask, batch_labels = batch_train[0].to(self.device), batch_train[1].to(self.device), batch_train[2].to(self.device)
                
                pooler = self.encoder(batch_input_ids, batch_attention_mask)
                pooler = pooler.detach().requires_grad_()

                with autocast():
                    loss = self.decoder(pooler, labels=batch_labels)
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
                    self.decoder.zero_grad()
                if step % 10 == 0:
                    self.decoder.eval()
                    batch_val_acc = []

                    for batch_val in val_dataloader:
                        with torch.no_grad():
                            batch_input_ids_val, batch_attention_mask_val, batch_labels_val = batch_val[0].to(self.device), batch_val[1].to(self.device), batch_val[2].to(self.device)

                            pooler = self.encoder(batch_input_ids_val, batch_attention_mask_val)
                            logits = self.decoder(pooler)
                            result = torch.argmax(logits, dim=-1)
                            batch_val_acc.append(accuracy_score(batch_labels_val.cpu().data.numpy(),result.cpu().data.numpy()))

                    avg_acc = sum(batch_val_acc) / len(batch_val_acc)
                    train_loss_record.append(loss*2)
                    val_acc_record.append(avg_acc)

                    logger.info("\nepoch %d, step %d, train_loss %.4f, val_acc %.4f" % (_epoch, step, loss*2, avg_acc))
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        early_num = 0
                        state_dict = self.decoder.state_dict()
                        torch.save(state_dict, "{}/{}".format(self.model_save_dir, self.decoder_ckpt_name))
                        # print("model saved")
                        logger.info("\nVal acc increased, decoder model saved as %s/%s. Epoch %d, Step %d, Train_loss %.4f, Val_acc %.4f" % (self.model_save_dir,self.decoder_ckpt_name,_epoch, step, loss*2, avg_acc))
                    else:
                        early_num += 1
                        if early_num == self.early_stop:
                            stop_tag = True
                            break
            if stop_tag:
                logger.info("\nEarly Stopping. Epoch %d, Step %d" % (_epoch, step))
                break
        
        return (train_loss_record, val_acc_record)


if __name__ == '__main__':
    encoder_name = "retrained_bert.bin"
    decoder_name = "classifier.bin"
    trainer = IntentTrainer(pretrained_model_dir=BASE_MODEL_PATH,\
                        model_save_dir=MODEL_PATH,\
                        encoder_ckpt_name=encoder_name,\
                        decoder_ckpt_name=decoder_name,\
                        use_cnn=False)
    batch_size = 16 
    epoch = 25
    logger.add(LOG_PATH+"classifier.log", rotation='10MB')
    logger.info("\nStart training with batch_size=%d, epoch=%d..." % (batch_size, epoch))
    trainer.train(TRAIN_PATH, VAL_PATH, batch_size=batch_size, epoch=epoch, early_stop=200)
    logger.info("\nTraining finished...")
    logger.remove()