import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import re
import unicodedata
from torch.nn.utils.rnn import pad_sequence
from .get_vocab import unicodeToAscii, normalizeString
from transformers import BertTokenizer, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

# focused classes, emotion
four_emo = ['anger', 'joy', 'sadness', 'neutral']
four_iem = ['angry', 'happy', 'sad', 'neutral']

rest_emo = ['surprise', 'disgust', 'fear', 'non-neutral']
# six_iem = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
six_iem = ['neu', 'fru', 'ang', 'hap', 'exc', 'sad']

class DialogDataset(data.Dataset):
    def __init__(self, path, class2id, word2id, key, roberta=False, pretrain_path = None, loss_mask = None):
        self.data = json.load(open(path))
        self.class2id = class2id
        self.word2id = word2id
        self.key = key
        
        self.loss_mask = loss_mask
        if loss_mask is not None:
            self.mask_data = self._get_mask(self.loss_mask)
        else:
            self.mask_data = None
        self.roberta = roberta
        if roberta and pretrain_path is not None:
            self.pad = 1
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
        else:
            self.pad = 0
    def _get_mask(self, path):
        with open(path) as f:
            data_text = f.readlines()
        mask = [line1.strip().split('\t')[1:] for line1 in data_text]
        mask = [[int(x) for x in a] for a in mask]
        return mask
    def __getitem__(self, index):
        item = self.data[index]
        
        # if self.key in ["ee", "er"]:
        #     label = {}
        #     label["ee"] = [self.class2id["ee"][x["ee"]] for x in item]
        #     label["er"] = [self.class2id["er"][x["er"]] for x in item]
        # else:
        label = [self.class2id[x[self.key]] for x in item]
        # speaker = [x["speaker"] for x in item]
        utterance = self.token2id(item)
        if self.mask_data is not None:
            mask = self.mask_data[index]
            return [utterance, label, mask]
        return [utterance, label]

    def token2id(self, utterance):
        idxs = []
        for text in utterance:
            text = normalizeString(text["utterance"])
            if self.roberta:
                idxs.append(self.roberta_(text))
                continue
            token = text.split()
            token = [x.lower() for x in token]
            idx = [self.word2id.get(x, 1) for x in token]
            idxs.append(idx)
        return idxs
    def roberta_(self, text):
        s = self.tokenizer.tokenize(text)
        s = ['<s>'] + s + ['</s>']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(s)
        return indexed_tokens

    def __len__(self):
        return len(self.data)
    # @staticmethod
    def collate_fn(self, data):
        if self.mask_data is not None:
            utterance, label, mask = data[0]
            mask = torch.tensor(mask).long()
        else:
            utterance, label = data[0]
        lens = [len(x) for x in utterance]
        new_lens = []
        for x in lens:
            if x < 1:
                x = 1
            new_lens.append(x)
        lens = np.array(new_lens)
        utterance = [torch.tensor(x) for x in utterance]
        pad_tensor = pad_sequence(utterance, batch_first=True, padding_value=self.pad)
        if isinstance(label, dict):
            for k,v in label.items():
                label[k] = torch.tensor(v).long()
        else:
            label = torch.tensor(label).long()
        if self.mask_data is not None:
            return [pad_tensor, label, lens, mask]
        else:
            return [pad_tensor, label, lens]

def get_dataloader(path, class2id, word2id ,key, args, shuffle=False, loss_mask =None):
    dataset = DialogDataset(path, class2id, word2id, key, loss_mask=loss_mask)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )

    return loader
def get_loss_mask(args, set_name, key):
    dataname = args.dataset
    if dataname in ["Persuasion"]:
        loss_mask = args.loss_mask + dataname + '/' + "{}_{}_{}_loss_mask.tsv".format(dataname.lower(), set_name, key)
    elif dataname in ["MultiWOZ"]:
        loss_mask = args.loss_mask + dataname + '/' + "{}_{}_loss_mask.tsv".format(dataname.lower(), set_name)
    else:
        loss_mask = None
    return loss_mask
def get_filed(key, args):
    dataname = args.dataset
    root = args.data_root
    word2id = json.load(open(args.vocab_root + "{}_vocab.json".format(dataname)))
    path1, path2, path3 = "{}/{}_train.json".format(dataname, dataname), "{}/{}_dev.json".format(dataname, dataname), "{}/{}_test.json".format(dataname, dataname)
    path1, path2, path3 = root + path1, root + path2, root + path3
    path = args.vocab_root
    if dataname in ["MELD", "MultiWOZ", "IEMOCAP"]:
        class2id = json.load(open(path+dataname+'class2id.json', encoding='utf-8'))
    elif dataname in ["DailyDialog", "Persuasion"]:
        class2id = json.load(open(path+dataname+'{}class2id.json'.format(key), encoding='utf-8'))
    # elif dataname in ["Persuasion"]:
    #     class2id_ee = json.load(open(path+dataname+'{}class2id.json'.format("ee"), encoding='utf-8'))
    #     class2id_er = json.load(open(path+dataname+'{}class2id.json'.format("er"), encoding='utf-8'))
        # class2id = {"ee": class2id_ee, "er":class2id_er}
    if key in ["emo"]: key = "emotion"
    # word2id = json.load(open("data/glove/glove_word2id.json")) persuasion_test_ee_loss_mask.tsv
    
    train = get_dataloader(path1, class2id, word2id, key,args, shuffle=True, loss_mask=get_loss_mask(args, "train", key))
    val = get_dataloader(path2, class2id, word2id, key, args, shuffle=False, loss_mask=get_loss_mask(args, "valid", key))
    test = get_dataloader(path3, class2id, word2id, key, args, shuffle=False, loss_mask=get_loss_mask(args, "test", key))
    return {'train':train, 'dev':val, 'test':test}

if __name__ == "__main__":
    path = "data/IEMOCAP/IEMOCAP_train.json"
    class2id = {k:v for v,k in enumerate(six_iem)}
    word2id = json.load(open("data/glove/glove_word2id.json"))
    ll = get_dataloader(path, class2id, word2id, 'emotion',1, True, 0)
    for i, data in enumerate(ll):
        print(data)