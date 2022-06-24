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

import json
from tqdm import tqdm
from nltk import sent_tokenize
import tensorflow_datasets as tfds
from transformers import BertTokenizer
from rouge import Rouge
rouge = Rouge()


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
tokenizer = BertTokenizer.from_pretrained('../dataset/BertModel/uncased_L-12_H-768_A-12', do_lower_case=True)
sent_maxlen=60
doc_maxlen=20

path = "data/newsroom/train.jsonl"
data = []
with open(path) as f:
    for ln in tqdm(f):
        obj = json.loads(ln)
        data.append(obj)

# newsroom_low = list(filter(lambda x: x['compression_bin']=='low', data))
newsroom = data[:100000]

def highlight_label(d):
    text_nlp = d['text'].split('\n\n')
    highlights = []
    for highlight in sent_tokenize(d['summary']):
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


dataset = []
for i in tqdm(range(len(newsroom))):
    data = dict()
    data['src_txt'] = newsroom[i]['text'].split('\n\n')
    data['tgt_txt'] = newsroom[i]['summary']
    tensor_highlights = np.zeros((doc_maxlen))
    labels = highlight_label(newsroom[i])
    for label in labels:
        if label<doc_maxlen:
            tensor_highlights[label]=1
    data['tgt'] = tensor_highlights
    sents = newsroom[i]['text'].split('\n\n')
    sents_tokened1 = tokenizer(sents, return_tensors='tf', padding=True)['input_ids']
    data['src_tensor'] = sents_tokened1
    sents_tokened2 = tokenizer(sents, return_tensors='np')['input_ids']
    src = []
    clss = [0]
    segs = []
    for i, tokend_sent in enumerate(sents_tokened2):
        src.extend(tokend_sent[:sent_maxlen])
        clss.append(len(src))
        if i%2==0:
            segs.extend([0]*len(tokend_sent[:sent_maxlen]))
        else:
            segs.extend([1]*len(tokend_sent[:sent_maxlen]))
    clss.pop()
    src = src[:512]
    src[-1] = 102
    data['src'] = src
    data['clss'] = clss[:doc_maxlen]
    data['segs'] = segs[:512]
    dataset.append(data)
    
import torch
torch.save(dataset, 'data/newsroom_dataset_all.pt')
print('save complete')