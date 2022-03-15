"""建立词表和glove预训练词向量"""
import time
import re
import json
import pickle
from tqdm import tqdm
import numpy as np
import unicodedata
import argparse

PAD = 0
UNK = 1
PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'

# Normalize strings
def unicodeToAscii(str):
	return ''.join(
		c for c in unicodedata.normalize('NFD', str)
		if unicodedata.category(c) != 'Mn'
	)


# Remove nonalphabetics
def normalizeString(str):
	str = unicodeToAscii(str.lower().strip())
	str = re.sub(r"([,.'!?])", r" \1", str)
	str = re.sub(r"[^a-zA-Z,.'!?]+", r" ", str)
	return str


# Read in scripts and labels from the dataset
def readUtterance(filename):
	with open(filename, encoding='utf-8') as data_file:
		data = json.loads(data_file.read())

	diadata = [[normalizeString(utter['utterance']) for utter in dialog] for dialog in data]
	# emodata = [[utter['emotion'] for utter in dialog] for dialog in data]
	
	return diadata

def load_txt_glove(filename, vocab, d_word_vec=300, save_path = None):
    """
    Loads 300x1 word vecs from Glove
    dtype: glove float64;
    """
    print('Initilaize with Glove 300d word vectors!')
    n_words = len(vocab)
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
    # initialize a numpy tensor
    embedding = np.random.uniform(-0.01, 0.01, (n_words, d_word_vec))
    for w, v in word_vecs.items():
        embedding[vocab[w]] = v

    # zero padding
    embedding[PAD] = np.zeros(d_word_vec)

    f = open(save_path, 'wb')
    pickle.dump(embedding, f)
    f.close()

    return embedding

class Dictionary:
	def __init__(self, name):
		self.name = name
		self.pre_word2count = {}
		self.rare = []
		self.word2count = {}
		self.word2index = {}
		self.index2word = {}
		self.n_words = 0
		self.max_length = 0
		self.max_dialog = 0

	# delete the rare words by the threshold min_count
	def delRare(self, min_count, padunk=True):

		# collect rare words
		for w,c in self.pre_word2count.items():
			if c < min_count:
				self.rare.append(w)

		# add pad and unk
		if padunk:
			self.word2index[PAD_WORD] = PAD
			self.index2word[PAD] = PAD_WORD
			self.word2count[PAD_WORD] = 1
			self.word2index[UNK_WORD] = UNK
			self.index2word[UNK] = UNK_WORD
			self.word2count[UNK_WORD] = 1
			self.n_words += 2

		# index words
		for w,c in self.pre_word2count.items():
			if w not in self.rare:
				self.word2count[w] = c
				self.word2index[w] = self.n_words
				self.index2word[self.n_words] = w
				self.n_words += 1

	def addSentence(self, sentence):
		sentsplit = sentence.split(' ')
		if len(sentsplit) > self.max_length:
			self.max_length = len(sentsplit)
		for word in sentsplit:
			self.addWord(word)

	def addWord(self, word):
		if word not in self.pre_word2count:
			self.pre_word2count[word] = 1
		else:
			self.pre_word2count[word] += 1

def get_vocab(paths, dataname, min_count=2, save_path = "/home/zjy/cp_dir/DialogGAT/data/"):
    print("Building vocab for dataset...")
    max_dialog = 0
    diadict = Dictionary(dataname)
    for path in paths:
        diadata  = readUtterance(path)
        for dia in diadata:
            if len(dia) > max_dialog: max_dialog = len(dia)
            for d in dia:
                diadict.addSentence(d)
    diadict.delRare(min_count=min_count, padunk=True)
    with open(save_path + "vocabs/{}_vocab.json".format(dataname), 'w', encoding="utf-8") as f:
        json.dump(diadict.word2index, f)
    print("build ok")
    return diadict.word2index
def generate_vocab_and_embedding(dataname, min_count):
	plist = ['_train', '_dev', '_test']
	paths = [datasets + dataname + x + '.json' for x in plist]
	dataname = dataname.split('/')[0]
	vocab = get_vocab(paths, dataname, min_count=min_count)
	save_path = "/home/zjy/cp_dir/DialogGAT/data/embeddings/{}_embedding.pkl".format(dataname)
	load_txt_glove(filename, vocab, save_path=save_path)

def get_class2id(dataname, savepath):
	plist = ['_train']
	path = [datasets + dataname + x + '.json' for x in plist]
	dataname = dataname.split('/')[0]
	with open(path[0], encoding='utf-8') as data_file:
		print("read {}".format(dataname))
		data = json.loads(data_file.read())
	
	def get_count_class(emodata):
		emodict = Dictionary(dataname)
		for emo in emodata:
			for e in emo:
				emodict.addSentence(e)
		emodict.delRare(min_count=0, padunk=False)
		return emodict.word2index, emodict.word2count
	def save_data(save_path, dataname, c2id, c2cocunt):
		with open(save_path+dataname+'class2id.json', 'w', encoding='utf-8') as f:
			json.dump(c2id, f)
		with open(save_path+dataname+'class2count.json', 'w', encoding='utf-8') as f:
			json.dump(c2cocunt, f)
		print("save {}".format(dataname))
	
	if dataname in ["IEMOCAP", "MELD"]:
		class_data = [[utter['emotion'] for utter in dialog] for dialog in data]
		class2id, class2count = get_count_class(class_data)
		save_data(savepath, dataname, class2id, class2count)
	elif dataname in ["MultiWOZ"]:
		class_data = [[utter['intent'] for utter in dialog] for dialog in data]
		class2id, class2count = get_count_class(class_data)
		save_data(savepath, dataname, class2id, class2count)
	elif dataname in ["Persuasion"]:
		class_data1 = [[utter['ee'] for utter in dialog] for dialog in data]
		class_data2 = [[utter['er'] for utter in dialog] for dialog in data]
		class2id1, class2count1 = get_count_class(class_data1)
		class2id2, class2count2 = get_count_class(class_data2)
		save_data(savepath, dataname+'ee', class2id1, class2count1)
		save_data(savepath, dataname+'er', class2id2, class2count2)
	elif dataname in ["DailyDialog"]:
		class_data1 = [[utter['emotion'] for utter in dialog] for dialog in data]
		class_data2 = [[utter['act'] for utter in dialog] for dialog in data]
		class2id1, class2count1 = get_count_class(class_data1)
		class2id2, class2count2 = get_count_class(class_data2)
		save_data(savepath, dataname+'emo', class2id1, class2count1)
		save_data(savepath, dataname+'act', class2id2, class2count2)


if __name__ == "__main__":
	
	# dataname = "IEMOCAP/IEMOCAP" #####
	#### IEMOCAP : mincount->2
	#### else -> 3
	filename = '/home/zjy/pre_word/glove.840B.300d.txt'
	datanames = ["IEMOCAP/IEMOCAP", "MELD/MELD", "DailyDialog/DailyDialog", "MultiWOZ/MultiWOZ", "Persuasion/Persuasion"]
	datasets = "/home/zjy/cp_dir/DialogGAT/data/"
	for dataname in datanames[3:4]:
		# get_class2id(dataname, datasets + "vocabs/")
		print(dataname)
		generate_vocab_and_embedding(dataname, min_count = 3)