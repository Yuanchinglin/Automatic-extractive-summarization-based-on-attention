import matplotlib as mpl
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from tqdm import tqdm
from nltk import sent_tokenize
import tensorflow_datasets as tfds
from rouge import Rouge
from transformers import TFBertModel, BertTokenizer, BertConfig, BertModel
import torch
rouge = Rouge()

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

configuration = BertConfig()
tokenizer = BertTokenizer.from_pretrained('../dataset/BertModel/uncased_L-12_H-768_A-12', do_lower_case=True)

import torch
dataset = torch.load(f'data/newsroom_dataset_all.pt')
sent_maxlen=60
doc_maxlen=20

from sklearn.model_selection import train_test_split
data_train, data_eval= train_test_split(dataset, test_size = 0.1)
len(data_train), len(data_eval)

src_text_train = [i['src_txt'] for i in dataset]
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
lxr = LexRank(src_text_train, stopwords=STOPWORDS['en'])

batch_size = 4
nbatch = len(data_train)//batch_size
train_batches = [data_train[i:i+batch_size] for i in range(nbatch)]

from collections import Counter
def make_adj(batch_data):
    adj_matrixs = []
    for i in range(len(batch_data)):
        src_txt = [e['src_txt'] for e in batch_data]
        tf_scores = [Counter(lxr.tokenize_sentence(sentence)) for sentence in src_txt[i]]
        adj = lxr._calculate_similarity_matrix(tf_scores)
        n = adj.shape[0]
        if doc_maxlen>=n:
            adj = tf.pad(adj, [[0, doc_maxlen - n], [0, doc_maxlen - n]])
        else:
            adj = adj[:doc_maxlen, :doc_maxlen]
        adj = tf.expand_dims(adj, 0)
        adj_matrixs.append(adj)
        adj_matrix = tf.concat(adj_matrixs, axis=0)
        adj_matrix = tf.cast(adj_matrix, tf.float32)
    return adj_matrix

def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data
def make_batch(data):
    pre_src = [x['src'] for x in data]
    pre_labels = [x['tgt'] for x in data]
    pre_segs = [x['segs'] for x in data]
    pre_clss = [x['clss'] for x in data]
    adj = make_adj(data)
    
    src = tf.convert_to_tensor(_pad(pre_src, 0))
    labels = tf.convert_to_tensor(pre_labels)
    segs = tf.convert_to_tensor(_pad(pre_segs, 0))
    mask = ~(src == 0)
    clss = np.array(_pad(pre_clss, -1, doc_maxlen))
    mask_cls = ~(clss == -1)
    clss[clss == -1] = 0
    return src, segs, clss, mask, mask_cls, adj, labels

class Bert(keras.layers.Layer):
    def __init__(self):
        super(Bert, self).__init__()
        self.model = TFBertModel(configuration)

    def call(self, x, segs, mask):
        output = self.model(x, attention_mask=mask, token_type_ids=segs, output_attentions=True)
        return output

class GCN(layers.Layer):
    def __init__(self, output_dim):
        super(GCN, self).__init__()
        self.fc = layers.Dense(output_dim)
    def call(self, x, adj):
        x = self.fc(x)
        out = tf.matmul(adj, x)
        out = tf.nn.relu(out)
        return out
    
class Classifier(keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert = Bert()
        self.fc1 = layers.Dense(128, kernel_initializer=tf.initializers.RandomUniform(), activation='sigmoid')
        self.fc2 = layers.Dense(128, kernel_initializer=tf.initializers.RandomUniform())
        self.fc3 = layers.Dense(1, kernel_initializer=tf.initializers.RandomUniform(), activation='sigmoid')
        self.gcn1 = GCN(400)
        self.gcn2 = GCN(150)
        # self.flatten = layers.Flatten()

    def call(self, x, segs, clss, mask, mask_cls, adj):
        output = self.bert(x, segs, mask)
        top_vecs = output[0]
        sents_vec = tf.stack([tf.gather(top_vec, cls, axis=0) for top_vec, cls in zip(top_vecs, clss)])
        sents_vec = sents_vec*mask_cls[..., tf.newaxis]
    
        out1 = self.gcn1(sents_vec, adj)
        # x *= mask[...,np.newaxis]
        out2 = self.gcn2(out1, adj)
        # x *= mask[...,np.newaxis]
        val1 = self.fc1(tf.concat([out2, sents_vec], axis=2))
        val2 = self.fc2(out2)
        out3 = tf.multiply(val1, val2)
        # out3 = tf.reduce_mean(out3, axis=1)
        out3 = tf.nn.relu(out3)
        # x = self.flatten(x)
        out = tf.squeeze(self.fc3(out3))
        return out*mask_cls
    
classifier = Classifier()
classifier.bert.trainable = False

optimizer = keras.optimizers.Adam()
loss_fcn = keras.losses.BinaryCrossentropy()

def train_step(src, segs, clss, mask, mask_cls, adj, labels):
    loss = 0
    with tf.GradientTape() as tape:
        logit = classifier(src, segs, clss, mask, mask_cls, adj)
        loss += loss_fcn(labels, logit)
  
    variable = classifier.trainable_variables
    gradients = tape.gradient(loss, variable)
    optimizer.apply_gradients(zip(gradients, variable))
    return tf.reduce_sum(loss)

nbatch_eval = len(data_eval)//batch_size
eval_batches = [data_eval[i:i+batch_size] for i in range(nbatch_eval)]

tf.random.set_seed=1968
epochs = 1
history_train = []
history_valid = []
for epoch in range(epochs):
    start = time.time()
    total_loss = 0
    for i, batch in enumerate(train_batches):
        src, segs, clss, mask, mask_cls, adj, labels = make_batch(batch)
        batch_loss = train_step(src, segs, clss, mask, mask_cls, adj, labels)
        total_loss += batch_loss
        if i % 5 == 0 and i > 0:
            cur_loss = total_loss / 5
            elapsed = time.time() - start
            src_, segs_, clss_, mask_, mask_cls_, adj_, labels_ = make_batch(eval_batches[3])
            out_valid = classifier(src_, segs_, clss_, mask_, mask_cls_, adj_)
            loss_valid = loss_fcn(labels_, out_valid)
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | valid {:5.2f}'.format(
                    epoch, i, nbatch,
                    elapsed,
                    cur_loss.numpy(), loss_valid))
            total_loss = 0
            start_time = time.time()

    # s = epoch%len(input_eval)//batch
    out_valid = classifier(src_, segs_, clss_, mask_, mask_cls_, adj_)
    loss_valid = loss_fcn(labels_, out_valid)

    history_train.append(total_loss/nbatch)
    history_valid.append(loss_valid)
    
    print('\n')
    print(f'Epoch{epoch} Loss:{total_loss/nbatch}')
    print(f'Loss Validation:{loss_valid}')
    print(f'Time {time.time()-start}')
    
R1 = []
R2 = []
RL = []
for j in tqdm(range(nbatch_eval)):
    src, segs, clss, mask, mask_cls, adj, labels = make_batch(eval_batches[j])
    pred = classifier(src, segs, clss, mask, mask_cls, adj)
    index = np.argsort(pred)[:, -2:]
    for e in range(len(index)):
        st = ''
        for i in index[e]:
            try:
                st += eval_batches[j][e]['src_txt'][i]
            except:
                pass
        score = rouge.get_scores([st],[eval_batches[j][e]['tgt_txt']])[0]
        R1.append(score['rouge-1']['f'])
        R2.append(score['rouge-2']['f'])
        RL.append(score['rouge-l']['f'])
        
r1_score = sum(R1)/len(R1)
r2_score = sum(R2)/len(R1)
rl_score = sum(RL)/len(R1)
print(f'R1:{r1_score}, R2:{r2_score}, RL:{rl_score}')