import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

from tqdm import tqdm
from nltk import sent_tokenize
import tensorflow_datasets as tfds
from transformers import BertTokenizer
from rouge import Rouge
rouge = Rouge()

tokenizer = BertTokenizer.from_pretrained('../dataset/BertModel/uncased_L-12_H-768_A-12', do_lower_case=True)

ds = tfds.load('cnn_dailymail', split='train', shuffle_files=True)
str_list = [(i['article'],i['highlights']) for i in iter(ds)]
cnn = [{'article':str(a.numpy(), encoding='utf8'), 'highlights':str(h.numpy(), encoding='utf8')} for (a,h) in str_list]
    

def highlight_label(d):
    text_nlp = sent_tokenize(d['article'])
    highlights = []
    for highlight in d['highlights'].split('\n'):
        try:
            scores = []
            for i, sent in enumerate(text_nlp):
                if len(sent) > 5:
                    rscores = rouge.get_scores([sent],[highlight])[0]['rouge-l']['f']
                    scores.append((i,rscores))
            highlight_id = np.argmax([scores[i][1] for i in range(len(scores))])
            highlight_sent_id = scores[highlight_id][0]
            highlights.append(highlight_sent_id)
        except:
            pass
    return highlights

sent_maxlen=80
doc_maxlen=60

dataset = []
for i in tqdm(range(len(cnn))):
    data = dict()
    data['src_txt'] = cnn[i]['article']
    data['tgt_txt'] = cnn[i]['highlights']
    tensor_highlights = np.zeros((doc_maxlen))
    labels = highlight_label(cnn[i])
    for label in labels:
        if label<doc_maxlen:
            tensor_highlights[label]=1
    data['tgt'] = tensor_highlights
    sents = sent_tokenize(cnn[i]['article'])
    sents_tokened = tokenizer(sents, return_tensors='tf', padding=True)['input_ids']
    data['src'] = sents_tokened
    dataset.append(data)
    
import torch
torch.save(dataset, 'data/cnndm_dataset.pt')

print("save complete")