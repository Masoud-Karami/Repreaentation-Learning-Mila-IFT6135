# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:17:13 2019

@author: karm2204
"""

import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# NOTE ==============================================
# This is where your models are imported
from models import GRU
from models import RNN
from models import make_model as TRANSFORMER

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data')
raw_data = ptb_raw_data(data_path='data')
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))



def repackage_hidden(h):

    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def avg_loss(model, data, model_name):
    model.eval()
    if model_name != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        BATCH_SIZE = model.batch_size
        SEQ_LEN = model.seq_len
    else:
        BATCH_SIZE = BATCH_SIZE_TRANSFORMER
        SEQ_LEN = SEQ_LEN_TRANSFORMER
        
    losses = np.empty((0, 35))
    
    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, BATCH_SIZE, SEQ_LEN)):
        if model_name == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        losses_batch = []
        for (output, label) in zip(outputs, targets):
            l = loss_fn(output, label)
            losses_batch.append(l.data.item())
        losses = np.vstack((losses, losses_batch))
    return np.mean(losses, axis=0)


print("\n########## Running Main Loop ##########################")

RNN_PATH = 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_9'
GRU_PATH = 'GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0'
TRANSFORMER_PATH = 'TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=.9_0'
EMB_SIZE_RNN = 200
HIDDEN_SIZE_RNN = 1500
SEQ_LEN_RNN = 35
BATCH_SIZE_RNN = 20
VOCAB_SIZE_RNN = 10000
NUM_LAYERS_RNN = 2
DP_KEEP_PROB_RNN = 0.9

EMB_SIZE_GRU = 200
HIDDEN_SIZE_GRU = 1500
SEQ_LEN_GRU = 35
BATCH_SIZE_GRU = 20
VOCAB_SIZE_GRU = 10000
NUM_LAYERS_GRU = 2
DP_KEEP_PROB_GRU = 0.35


HIDDEN_SIZE_TRANSFORMER = 512
VOCAB_SIZE_TRANSFORMER = 10000
NUM_LAYERS_TRANSFORMER = 2
DP_KEEP_PROB_TRANSFORMER = 0.9
BATCH_SIZE_TRANSFORMER = 128
SEQ_LEN_TRANSFORMER = 35


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

word_to_id, id_2_word = _build_vocab(os.path.join('data', 'ptb' + ".train.txt"))

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()



model_classes = [RNN, GRU, TRANSFORMER]
for model_class in model_classes:
    if model_class == RNN:
        MODEL_PATH = RNN_PATH
        model = model_class(emb_size=EMB_SIZE_RNN,
                    hidden_size=HIDDEN_SIZE_RNN,
                    seq_len=SEQ_LEN_RNN,
                    batch_size=BATCH_SIZE_RNN,
                    vocab_size=VOCAB_SIZE_RNN,
                    num_layers=NUM_LAYERS_RNN,
                    dp_keep_prob=DP_KEEP_PROB_RNN)
        model_name = 'RNN'

    if model_class == GRU:
        MODEL_PATH = GRU_PATH
        model = model_class(emb_size=EMB_SIZE_GRU,
                    hidden_size=HIDDEN_SIZE_GRU,
                    seq_len=SEQ_LEN_GRU,
                    batch_size=BATCH_SIZE_GRU,
                    vocab_size=VOCAB_SIZE_GRU,
                    num_layers=NUM_LAYERS_GRU,
                    dp_keep_prob=DP_KEEP_PROB_GRU)
        model_name = 'GRU'

    if model_class == TRANSFORMER:
        MODEL_PATH = TRANSFORMER_PATH
        model = model_class(vocab_size=VOCAB_SIZE_TRANSFORMER, n_units=HIDDEN_SIZE_TRANSFORMER,
                            n_blocks=NUM_LAYERS_TRANSFORMER, dropout=1.-DP_KEEP_PROB_TRANSFORMER)
        model_name = 'TRANSFORMER'
    load_path = os.path.join(MODEL_PATH, 'best_params.pt')
    model.load_state_dict(torch.load(load_path))

    if model_class != TRANSFORMER:
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    model = model.to(device)
    loss_array = avg_loss(model, valid_data, model_name)
    plt.plot(loss_array, '-o', label=model_name)
plt.title("Validation Loss Vs. Timesteps")
plt.ylabel("Validation Loss")
plt.xlabel("Timestep")
plt.legend()
plt.savefig("Q5.1_PLOT.jpg")