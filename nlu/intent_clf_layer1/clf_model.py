# -*- coding:utf-8 -*-

import os
import pickle
import numpy as np


class CLFModel(object):
    def __init__(self, model_save_path):
        super(CLFModel, self).__init__()
        self.model_save_path = model_save_path
        self.id2label = pickle.load(open(os.path.join(self.model_save_path,'id2label.pkl'),'rb'))
        self.vec = pickle.load(open(os.path.join(self.model_save_path,'vec.pkl'),'rb'))
        self.LR_clf = pickle.load(open(os.path.join(self.model_save_path,'LR.pkl'),'rb'))
        self.gbdt_clf = pickle.load(open(os.path.join(self.model_save_path,'gbdt.pkl'),'rb'))
        self.svc_clf = pickle.load(open(os.path.join(self.model_save_path,'svc.pkl'),'rb'))

    def predict(self,text):
        text = ' '.join(list(text.lower()))
        text = self.vec.transform([text])
        proba1 = self.LR_clf.predict_proba(text)
        proba2 = self.gbdt_clf.predict_proba(text)
        proba3 = self.svc_clf.predict_proba(text)
        label = np.argmax((proba1+proba2+proba3)/3, axis=1)
        return self.id2label.get(label[0])


def get_CLFModel():
    return CLFModel('D:/Desktop/nlp/KBQA/KBQA-main/nlu/intent_clf_layer1/model_file/')


if __name__ == '__main__':
    model = CLFModel('./model_file/')

    text='医生你好'
    label = model.predict(text)
    print(label)