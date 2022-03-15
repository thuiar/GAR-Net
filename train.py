""" Train """
import os, json, pickle
import argparse
import Utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.GAR import GAR
from kits import emotrain, emoeval
from datetime import datetime

from config import Config
import math
import time


def main():
    args = Config() # 加载参数
    Utils.setup_seed(args.seed)
    root = args.data_root #数据集存放的根目录
    # Load data field
    print("Loading field...")
    dataset = args.dataset
    key = args.key
    wr = args.wr
    # args.dataset = dataset
    if args.multi:
        # from data.dataloaders import get_filed
        from data.dataloader_for_tav import get_filed # 多模态
    else:
        from data.dataloaders import get_filed

    field = get_filed(key=key, args= args)
    if dataset in ["Persuasion", "DailyDialog"]:
        wc_dataset = dataset + key
    else:
        wc_dataset = dataset
    def get_weight_and_wc(wc_dataset, wr):
        word2count = json.load(open(args.vocab_root + "{}class2count.json".format(wc_dataset)))
        keys_list = list(word2count.keys())
        # keys_list = [k for k in keys_list if k != "0"]
        weights = Utils.get_weights(word2count, keys_list, weight_rate=wr)
        return word2count, weights
    word2count, weights = get_weight_and_wc(wc_dataset, wr)
    args.num_classes=len(word2count)
    
    print("Initializing word embeddings...")
    if args.roberta:
        embedding = Utils.RobertaEmbedding(path = args.data_root + "roberta-base")
        for param in embedding.named_parameters():
            param[1].requires_grad = args.embedding_train
        d_word_vec = 768
    else:
        np_embedding = pickle.load(open(args.embedding_root + "{}_embedding.pkl".format(dataset), 'rb'))
        embedding = nn.Embedding(np_embedding.shape[0], np_embedding.shape[1], padding_idx=0)
        embedding.weight.data.copy_(torch.from_numpy(np_embedding))
        embedding.weight.requires_grad = args.embedding_train
        d_word_vec = 300

    model = GAR(args, embedding=embedding)

    emotrain(model=model,
            data_loader=field,
            args=args,
            weights=weights,
            dataname=dataset,
            key=key)

    # Load the best model to test
    print("Load best models for testing!")
    # model = Utils.model_loader(args.save_dir, args.type, args.dataset)
    model_path = '{}/DGAT_{}.pt'.format(args.save_dir, args.dataset)
    test_loader = field['test']
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    pAccs = emoeval(model=model,
                    data_loader=test_loader,
                    args=args,
                    weights=weights,
                    dataname=dataset,
                    key=key)
    print("Test: wr:{} ACCs-WA-UWA {}".format(wr, pAccs))

    # Save the test results
    record_file = '{}/GAR-Net_{}_{}_wr.txt'.format(args.save_dir, args.dataset, key)
    if os.path.isfile(record_file):
        f_rec = open(record_file, "a")
    else:
        f_rec = open(record_file, "w")
    f_rec.write("{} - {} - {}- {} -{}\t{}_wr:{}:\t{}\n".format(datetime.now(), args.d_h1, args.d_h2, args.ll, args.drop, args.lr, wr, pAccs))
    f_rec.close()


if __name__ == '__main__':
    torch.cuda.set_device(1)
    main()


