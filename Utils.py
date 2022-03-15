"""
Utils

"""
import json
import pickle
import torch
import os
import math
import random
import numpy as np
import time
from transformers import BertTokenizer, RobertaModel

class RobertaEmbedding(torch.nn.Module):
    def __init__(self, path="data/roberta-base"):
        super(RobertaEmbedding, self).__init__()
        self.roberta = RobertaModel.from_pretrained(path)
    def forward(self, x, lens):
        mask = self.to_mask(lens, x.size(1))
        x,_ = self.roberta(x, attention_mask=mask) #, attention_mask=mask
        return x
    def to_mask(self, lens, b):
        mask = []
        for i in range(len(lens)):
            l = lens[i].item()
            att_mask = torch.zeros(b).long()  # (1, L)
            att_mask = att_mask.cuda() if torch.cuda.is_available() else att_mask
            att_mask[:l] = 1
            mask.append(att_mask)
        return torch.stack(mask, 0)

def loss_weight(word2count, focus_dict, rate=1.0):
    """ Loss weights """
    min_emo = float(min([word2count[w] for w in focus_dict]))
    weight = [math.pow(min_emo / word2count[k], rate) if k in focus_dict
              else 0.0001 for k,v in word2count.items()]
    weight = np.array(weight)
    weight /= np.sum(weight)

    return weight

def get_weights(word2count, focus_emo, weight_rate):
    weights = torch.from_numpy(loss_weight(word2count, focus_emo, rate=weight_rate)).float()
    print(focus_emo)
    print(weights)
    return weights

# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Timer
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# model saver
def model_saver(model, path, module, dataset):
    if not os.path.isdir(path):
        os.makedirs(path)
    model_path = '{}/{}_{}.pt'.format(path, module, dataset)
    torch.save(model, model_path)

# model loader
def model_loader(path, module, dataset):
    model_path = '{}/{}_{}.pt'.format(path, module, dataset)
    model = torch.load(model_path, map_location='cpu')
    return model


def saveToJson(path, object):
    t = json.dumps(object, indent=4)
    f = open(path, 'w')
    f.write(t)
    f.close()

    return 1


def saveToPickle(path, object):
    file = open(path, 'wb')
    pickle.dump(object, file)
    file.close()

    return 1


def loadFrPickle(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj


def load_bin_vec(filename, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    dtype: word2vec float32, glove float64;
    Word2vec's input is encoded in UTF-8, but output is encoded in ISO-8859-1
    """
    print('Initilaize with Word2vec 300d word vectors!')
    word_vecs = {}
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split()[0:2])
        binary_len = np.dtype('float32').itemsize * layer1_size
        num_tobe_assigned = 0
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('iso-8859-1')
                if ch == ' ':
                    word = ''.join(word)
                    #print(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                vector = np.fromstring(f.read(binary_len), dtype='float32')
                word_vecs[word] = vector / np.sqrt(sum(vector**2))
                num_tobe_assigned += 1
            else:
                f.read(binary_len)
        print("Found words {} in {}".format(vocab_size, filename))
        match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
        print("Matched words {}, matching rate {} %".format(num_tobe_assigned, match_rate))
    return word_vecs


def load_txt_glove(filename, vocab):
    """
    Loads 300x1 word vecs from Glove
    dtype: glove float64;
    """
    print('Initilaize with Glove 300d word vectors!')
    word_vecs = {}
    vector_size = 300
    with open(filename, "r") as f:
        vocab_size = 0
        num_tobe_assigned = 0
        for line in f:
            vocab_size += 1
            splitline = line.split()
            word = " ".join(splitline[0:len(splitline) - vector_size])
            if word in vocab:
                vector = np.array([float(val) for val in splitline[-vector_size:]])
                word_vecs[word] = vector / np.sqrt(sum(vector**2))
                num_tobe_assigned += 1

        print("Found words {} in {}".format(vocab_size, filename))
        match_rate = round(num_tobe_assigned/len(vocab)*100, 2)
        print("Matched words {}, matching rate {} %".format(num_tobe_assigned, match_rate))
    return word_vecs


def load_pretrain(d_word_vec, diadict, type='word2vec'):
    """ initialize nn.Embedding with pretrained """
    if type == 'word2vec':
        filename = 'word2vec300.bin'
        word2vec = load_bin_vec(filename, diadict.word2index)
    elif type == 'glove':
        filename = 'pre_word/glove.840B.300d.txt'
        word2vec = load_txt_glove(filename, diadict.word2index)

    # initialize a numpy tensor
    embedding = np.random.uniform(-0.01, 0.01, (diadict.n_words, d_word_vec))
    for w, v in word2vec.items():
        embedding[diadict.word2index[w]] = v

    # zero padding
    embedding[Const.PAD] = np.zeros(d_word_vec)

    return embedding
