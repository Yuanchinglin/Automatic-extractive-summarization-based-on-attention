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

import pickle
with open('data/multi_news.pkl', 'rb') as pkl_file:
    multi_news = pickle.load(pkl_file)
    
docs = [d['document'].split('\n  \n') for d in tqdm(multi_news)]

def summary_label(d):
    text_nlp = d['document'].split('\n  \n')
    summaries = []
    for summary in sent_tokenize(d['summary']):
        try:
            scores = []
            for i, sent in enumerate(text_nlp):
                if len(sent) > 5:
                    rscores = rouge.get_scores([sent],[summary])[0]['rouge-2']['f']
                    scores.append((i,rscores))
            summary_id = np.argmax([scores[i][1] for i in range(len(scores))])
            summary_sent_id = scores[summary_id][0]
            summaries.append(summary_sent_id)
        except:
            pass
    return summaries

sent_maxlen=50
doc_maxlen=60
batch = 32

dataset = []
for i in tqdm(range(len(multi_news))):
    data = dict()
    data['src_txt'] = docs[i]
    data['tgt_txt'] = multi_news[i]['summary']
    tensor_highlights = np.zeros((doc_maxlen))
    labels = summary_label(multi_news[i])
    for label in labels:
        if label<doc_maxlen:
            tensor_highlights[label]=1
    data['tgt'] = tensor_highlights
    sents_tokened = tokenizer(docs[i], return_tensors='tf', padding=True)['input_ids']
    data['src'] = sents_tokened
    dataset.append(data)
    
import torch
torch.save(dataset, 'data/multi_news_dataset.pt')

print("save complete")