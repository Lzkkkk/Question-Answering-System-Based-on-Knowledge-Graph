import torch
import torch.nn.functional as F

BASE_MODEL_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/pretrained_models/chinese_roberta_wwm_ext_pytorch"
MODEL_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/model"
TRAIN_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/data/train.csv"
VAL_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/data/val.csv"
TEST_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/data/test.csv"
LABEL_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/data/label"
LOG_PATH = "D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer2/log"

def l2_norm(input):
    return input/(input**2).sum(dim=-1,keepdim=True).sqrt()


def get_pair_labels(labels):
    return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()


def contrastive_mse_loss(logits, labels):
    logits_norm = l2_norm(logits)  # (batch_size, num_labels)
    batch_sims = torch.mm(logits_norm, logits_norm.T)  # (batch_size, batch_size)
    return (F.mse_loss(batch_sims, get_pair_labels(labels))*len(labels)) / (len(labels)-1)


def infoNCE_loss(logits, labels, temperature=0.05):
    # version 1
    # logits_norm = l2_norm(logits)
    # batch_sims = torch.mm(logits_norm, logits_norm.T)
    # eye_mask = (torch.eye(len(labels))*(-1e12)).cuda()
    # batch_sims += eye_mask
    # batch_sims /= temperature
    # pair_labels = get_pair_labels(labels).cuda()
    # denominator = batch_sims.exp().sum(dim=-1, keepdim=True)
    # numerator = (batch_sims.exp()*pair_labels).sum(dim=-1, keepdim=True)
    # loss = numerator/denominator
    # loss *= (-1 / (pair_labels.sum(dim=-1,keepdim=True)-1))
    # loss = loss.log()

    # version 2
    logits_norm = l2_norm(logits)
    batch_sims = torch.mm(logits_norm, logits_norm.T)
    batch_sims /= temperature

    sims_max, _ = torch.max(batch_sims, dim=1, keepdim=True)
    batch_sims -= sims_max.detach()

    pair_labels = get_pair_labels(labels).cuda()
    eye_mask = (torch.eye(len(labels))).cuda()
    pair_labels -= eye_mask
    denominator = batch_sims.exp().sum(dim=-1, keepdim=True)
    numerator = batch_sims.exp()
    loss = (((numerator/denominator).log())*pair_labels).sum(dim=-1, keepdim=True)
    loss *= (-1 / (pair_labels.sum(dim=-1,keepdim=True)))
    
    return loss.mean()