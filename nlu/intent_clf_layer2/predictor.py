import torch
import math
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report
from .utils import BASE_MODEL_PATH,MODEL_PATH,TEST_PATH,LABEL_PATH,LOG_PATH
from .models import MyBert, Classifier
from loguru import logger
from transformers import logging


class IntentPredictor(object):
    def __init__(self, use_cuda=False, pretrained_model_dir=None,
                 model_dir=None, encoder_ckpt_name=None, decoder_ckpt_name=None, label_dir=None, use_cnn=True):
        self.pretrained_model_dir = pretrained_model_dir
        self.encoder_dir = model_dir + '/' + encoder_ckpt_name
        self.decoder_dir = model_dir + '/' + decoder_ckpt_name
        self.use_cuda = use_cuda
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.targets_names = [line.strip() for line in open(label_dir, 'r', encoding="utf-8")]
        self.id2label = {idx: label for idx, label in enumerate(self.targets_names)}
        self.label2id = {label: idx for idx, label in enumerate(self.targets_names)}

        self.encoder = MyBert(self.pretrained_model_dir)
        self.decoder = Classifier(use_cnn=self.use_cnn)
        self.tokenizer = self.encoder.bert_tokenizer
        self.encoder.load_state_dict(torch.load(self.encoder_dir, map_location=self.device))
        self.decoder.load_state_dict(torch.load(self.decoder_dir, map_location=self.device))
        self.encoder.eval()
        self.decoder.eval()
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def predict(self, test_path, batch_size=32, max_len=512):
        test_data = pd.read_csv(test_path)
        texts = test_data["text"]
        labels = test_data["label"].tolist()
        
        results = []

        for batch_idx in range(math.ceil(len(texts) / batch_size)):
            text_batch = texts[batch_idx*batch_size: batch_size*(batch_idx+1)]
            batch_max_len = min(max([len(text) for text in text_batch]) + 2, max_len)

            batch_input_ids, batch_att_mask = [], []
            for text in text_batch:
                assert isinstance(text, str)
                encoded_dict = self.tokenizer.encode_plus(text, max_length=batch_max_len, padding="max_length",
                                                          return_tensors="pt", truncation=True)
                batch_input_ids.append(encoded_dict["input_ids"])
                batch_att_mask.append(encoded_dict["attention_mask"])
            batch_input_ids = torch.cat(batch_input_ids).to(self.device)
            batch_att_mask = torch.cat(batch_att_mask).to(self.device)

            with torch.no_grad():
                pooler = self.encoder(batch_input_ids, batch_att_mask)
                logits = self.decoder(pooler)
                result = torch.argmax(logits, dim=-1).cpu().data.numpy()
                results.extend(result.tolist())

        # print(classification_report(labels, results, target_names=self.targets_names))
        logger.info("\n"+classification_report(labels, results, target_names=self.targets_names, digits=4))

    def online_pred(self, text, max_length=512):
        # print(text)
        max_length = min(len(text)+2, max_length)
        input_ids, att_mask = [], []
        encoded_dict = self.tokenizer.encode_plus(text, max_length=max_length+2, padding="max_length",
                                                  return_tensors="pt", truncation=True)
        input_ids.append(encoded_dict["input_ids"])
        att_mask.append(encoded_dict["attention_mask"])
        batch_input_ids = torch.cat(input_ids).to(self.device)
        batch_att_mask = torch.cat(att_mask).to(self.device)

        with torch.no_grad():
            pooler = self.encoder(batch_input_ids, batch_att_mask)
            logits = self.decoder(pooler)
            logits = nn.Softmax(dim=-1)(logits)
            confidence, idx = torch.max(logits, dim=1)
            
            # logger.info("\nInput text: %s, prediction: %s, confidence: %f"%(text,self.id2label[idx.data.item()],confidence.data.item()))
            return {"result": self.id2label[idx.data.item()], "confidence": confidence.data.item()}



def get_IntentPredictor():
    logging.set_verbosity_error()
    return IntentPredictor(pretrained_model_dir=BASE_MODEL_PATH, model_dir=MODEL_PATH, label_dir=LABEL_PATH,\
                        encoder_ckpt_name="retrained_bert.bin",\
                        decoder_ckpt_name="classifier.bin",\
                        use_cuda=True, use_cnn=False)


if __name__ == '__main__':
    logger.add(LOG_PATH+"/predictor.log", rotation='10MB')
    pred = IntentPredictor(pretrained_model_dir=BASE_MODEL_PATH, model_dir=MODEL_PATH, label_dir=LABEL_PATH,\
                        encoder_ckpt_name="retrained_bert.bin",\
                        decoder_ckpt_name="classifier.bin",\
                        use_cuda=True, use_cnn=False)
    pred.predict(TEST_PATH)
    # res = pred.online_pred("我朋友得了白血病，还有救么")
    # print(res)
    logger.remove()
