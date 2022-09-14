import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from .utils import infoNCE_loss


# refer to: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
class TextCNN(nn.Module):
    def __init__(self, hidden_size, kernel_num, kernel_sizes):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, hidden_size)) for K in kernel_sizes])

    def forward(self, hidden_states):
        # (N,Ci,W,D)
        hidden_states = hidden_states.unsqueeze(1)
        # [(N, Co, W), ...]*len(Ks)
        hidden_states = [F.relu(conv(hidden_states)).squeeze(3) for conv in self.convs]

        # [(N, Co), ...]*len(Ks)
        hidden_states = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in hidden_states]

        hidden_states = torch.cat(hidden_states, 1)

        return hidden_states


class BertTextcnn(nn.Module):
    def __init__(self, bert_base_path, class_nums=13, output_channels=256, hidden_size=768, use_cnn=True):
        super(BertTextcnn, self).__init__()

        self.bert_config = BertConfig.from_json_file(bert_base_path+"/bert_config.json")
        self.bert_model = BertModel.from_pretrained(bert_base_path,config=self.bert_config)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        self.use_cnn = use_cnn

        if self.use_cnn == False:
            self.classifier = nn.Linear(hidden_size, class_nums)
        else:
            self.classifier = nn.Sequential(
                TextCNN(hidden_size, kernel_num=output_channels, kernel_sizes=(2, 3, 4)),
                nn.Linear(output_channels * 3, class_nums)
            )

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,).last_hidden_state

        if self.use_cnn == False:
            hidden_states = hidden_states[:, 0, :]  # cls_embed (batchSize, hidden_nums|768)
        
        logits = self.classifier(hidden_states) # logits (batchSize, num_labels)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def tokenizer(self):
        return self.bert_tokenizer


class MyBert(nn.Module):
    def __init__(self, bert_base_path):
        super(MyBert, self).__init__()

        self.bert_config = BertConfig.from_json_file(bert_base_path+"/bert_config.json")
        self.bert_model = BertModel.from_pretrained(bert_base_path,config=self.bert_config)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_path)


    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states, pooler = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,)[:2]
        

        if labels is not None:
            infonce_loss = infoNCE_loss(pooler, labels, temperature=0.05)

            return infonce_loss

        return pooler

    def tokenizer(self):
        return self.bert_tokenizer


class Classifier(nn.Module):
    def __init__(self, class_nums=13, output_channels=256, hidden_size=768, use_cnn=True):
        super(Classifier, self).__init__()
        self.use_cnn = use_cnn

        if self.use_cnn == False:
            self.classifier = nn.Linear(hidden_size, class_nums)
        else:
            self.classifier = nn.Sequential(
                TextCNN(hidden_size, kernel_num=output_channels, kernel_sizes=(3, 4, 5)),
                nn.Linear(output_channels * 3, class_nums)
            )

    def forward(self, pooler, labels=None):
        logits = self.classifier(pooler) # logits (batchSize, num_labels)

        if labels is not None:
            # cross entropy loss
            return F.cross_entropy(logits, labels)

        return logits

    def tokenizer(self):
        return self.bert_tokenizer

