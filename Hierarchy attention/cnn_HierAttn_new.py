import matplotlib as mpl
import matplotlib.pyplot as plt
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
import nltk

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

configuration = BertConfig()
bertmodel = TFBertModel(configuration)
tokenizer = BertTokenizer.from_pretrained('../dataset/BertModel/uncased_L-12_H-768_A-12', do_lower_case=True)
sent_maxlen=80
doc_maxlen=60
batch = 32

dataset = torch.load('data/cnndm_dataset.pt')
Rdataset = filter(lambda x: x['src'].shape[0]<doc_maxlen, dataset)
Rdataset = list(Rdataset)
Rdataset = filter(lambda x: x['src'].shape[1]<sent_maxlen, Rdataset)
Rdataset = list(Rdataset)
print(len(Rdataset))

from sklearn.model_selection import train_test_split
data_train, data_eval= train_test_split(Rdataset, test_size = 0.1)

from util import scaled_dot_product_attention, MultiHeadAttention, feed_forward

def create_embedding_mask(batch_data):
    mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    return mask

class SentEncoder(keras.layers.Layer):
  # x -> multihead attention->add & normal & dropout-> feed forward-> add & norm & dropout
  def __init__(self, input_vocab_size, d_model, dff, rate=0.2):
    super(SentEncoder, self).__init__()
    self.d_model = d_model
    self.embedding = keras.layers.Embedding(input_vocab_size, self.d_model)
    self.mha_sent = MultiHeadAttention(d_model)

    self.ffn = feed_forward(d_model, dff)
    self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = keras.layers.Dropout(rate)
    self.dropout2 = keras.layers.Dropout(rate)
#     self.fc = keras.layers.Dense((1), activation='sigmoid')

  def call(self, x, training, encoder_padding_mask):
    # x.shape: [batch, doc_len, sent_len]
    #     ->[batch, doc_len, sent_len, d_model]

    x = self.embedding(x)
    # attention_output: [batch, doc_len, sent_len, d_model]
    mask_inverse = tf.cast(~tf.cast(encoder_padding_mask, tf.bool), tf.float32)
    attn_output, _ = self.mha_sent(x, x, x, encoder_padding_mask[..., tf.newaxis])

    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.norm1(x + attn_output)

    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    # out2: [batch, doc_len, sent_len, d_model]
    out2 = self.norm2(out1 + ffn_output)
    # sent_attnetion: [batch, doc_len, sent_len]
    sent_attnetion = tf.reduce_mean(out2, axis=-1)*mask_inverse
    return sent_attnetion

class DocEncoder(keras.layers.Layer):
  def __init__(self, d_model, dff, rate=0.2):
    super(DocEncoder, self).__init__()
    self.ffn2 = feed_forward(d_model, dff)
    self.dropout1 = keras.layers.Dropout(rate)
    self.dropout2 = keras.layers.Dropout(rate)
    self.mha_doc = MultiHeadAttention(d_model)
    self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.fc2 = keras.layers.Dense((1), activation='sigmoid')
  
  def call(self, x, training, encoder_padding_mask):
    # x: [batch, doc_len, sent_len]
    mask_inverse = tf.cast(~tf.cast(encoder_padding_mask, tf.bool), tf.float32)
    attn_output2, sent_attn = self.mha_doc(x, x, x, encoder_padding_mask[:,:,0, tf.newaxis])
    attn_output2 = self.dropout1(attn_output2, training=training)
    # ffn_output2: [batch, doc_len, d_model]
    ffn_output2 = self.ffn2(attn_output2)
    ffn_output2 = self.dropout2(ffn_output2, training=training)
    out3 = self.norm3(attn_output2 + ffn_output2) 
    out3 = out3 * mask_inverse[:,:,0, tf.newaxis]
    predict = tf.squeeze(self.fc2(out3)) * mask_inverse[:,:,0]
    return predict, sent_attn

class Encoder(keras.Model):
  def __init__(self, input_vocab_size, d_model, dff, rate=0.2):
    super(Encoder, self).__init__()
    self.sent_encoder = SentEncoder(input_vocab_size, d_model, dff)
    self.doc_encoder = DocEncoder(d_model, dff)

  def call(self, x, training, encoder_paddding_mask):
    out = self.sent_encoder(x, training, encoder_paddding_mask)
    predict, sent_attn = self.doc_encoder(
        out, training, encoder_paddding_mask)
    return predict, sent_attn

inp_vocab_size = tokenizer.vocab_size
encoder = Encoder(inp_vocab_size, 128, 128)

src_train = [data_train[i]['src'] for i in range(len(data_train))]
input_train = tf.ragged.stack(src_train).to_tensor(shape=(len(data_train), doc_maxlen, sent_maxlen))
src_eval = [data_eval[i]['src'] for i in range(len(data_eval))]
input_eval = tf.ragged.stack(src_eval).to_tensor(shape=(len(data_eval), doc_maxlen, sent_maxlen))
output_train = tf.ragged.stack([data_train[i]['tgt'] for i in range(len(data_train))]).to_tensor()
output_eval = tf.ragged.stack([data_eval[i]['tgt'] for i in range(len(data_eval))]).to_tensor()

optimizer = keras.optimizers.Adam(lr=0.01, epsilon=1e-9)
loss_fcn = keras.losses.BinaryCrossentropy()

@tf.function
def train_step(inp, targ):
    loss = 0
    encoding_padding_mask = create_embedding_mask(inp)
    with tf.GradientTape() as tape:
        logit, _ = encoder(inp, True, encoding_padding_mask)
        loss += loss_fcn(targ, logit)
  
    variable = encoder.trainable_variables
    gradients = tape.gradient(loss, variable)
    optimizer.apply_gradients(zip(gradients, variable))
    return tf.reduce_sum(loss)

epochs = 10
history_train = []
history_valid = []
for epoch in range(epochs):
    start = time.time()
    total_loss = 0
    step_per_train = len(input_train)//batch
    
    s = epoch%len(input_eval)//batch
    inp_valid = input_eval[s*batch:s*batch+batch]
    targ_valid = output_eval[s*batch:s*batch+batch]
    inp_valid = tf.convert_to_tensor(inp_valid)
    
    for step in range(step_per_train):
        inp = input_train[step*batch:step*batch+batch]
        targ = output_train[step*batch:step*batch+batch]
        inp = tf.convert_to_tensor(inp)
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
        if step%100==0:
#             history_train.append(batch_loss)
            mask_valid = create_embedding_mask(inp_valid)
            output_valid, _ = encoder(inp_valid, False, mask_valid)
            loss_valid = loss_fcn(targ_valid, output_valid)
#             logger.info(f'{epoch},{step},{batch_loss.numpy()},{loss_valid.numpy()},{time.time()-start}')
            print(epoch, step, batch_loss.numpy(), loss_valid.numpy(), time.time()-start)
    history_valid.append(loss_valid)
    history_train.append(total_loss/step_per_train)
    print('\n')
    print(f'Epoch{epoch} Loss:{total_loss/step_per_train}')
    print(f'Loss Validation:{loss_valid}')
    print(f'Time {time.time()-start}')
    
mask_eval = create_embedding_mask(input_eval)
output, sent_attn = encoder(input_eval[:300], False, mask_eval[:300])
index = tf.argsort(output)[:, -3:]

R1 = []
R2 = []
RL = []
for e in range(300):
  st = ''
  for i in index[e]:
        try:
            st += sent_tokenize(data_eval[e]['src_txt'])[i]
        except:
            pass
  score = rouge.get_scores([st],[data_eval[e]['tgt_txt']])[0]
  R1.append(score['rouge-1']['f'])
  R2.append(score['rouge-2']['f'])
  RL.append(score['rouge-l']['f'])
    
r1_score = sum(R1)/len(R1)
r2_score = sum(R2)/len(R1)
rl_score = sum(RL)/len(R1)
print(r1_score, r2_score, rl_score)