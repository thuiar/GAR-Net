"""
Train on Emotion dataset
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Utils
import math
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import AdamW

def emotrain(model, data_loader, args, weights=None, dataname = "IEMOCAP", key = None):
    # start time
    time_st = time.time()
    decay_rate = args.decay

    # Load in the training set and validation set
    train_loader = data_loader['train']
    dev_loader = data_loader['dev']

    # Optimizer
    # lr = args.lr
    if args.roberta:
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        model_opt = AdamW(parameters_to_optimize, lr=args.lr, correct_bias=False)
    else:
        model_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    over_fitting = 0
    cur_best = -1e10
    glob_steps = 0
    report_loss = 0

    for epoch in range(1, args.epochs + 1):
        model_opt.param_groups[0]['lr'] *= decay_rate	# Decay the lr every epoch
        print("===========Epoch {}==============".format(epoch))
        print("-{}-{}".format(epoch, Utils.timeSince(time_st)))
        
        for data in tqdm(train_loader):
            # Tensorize a dialogue, a dialogue is a batch
            feat = data[0]
            label = data[1]
            lens = data[2]
            
            if len(data) == 4:
                mask = data[3]
            else:
                mask = None

            if args.gpu != None:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
                device = torch.device(0)
                model.cuda(device)
                feat = feat.cuda(device)
                label = label.cuda(device)
                if weights is not None:
                    weights = weights.cuda(device)
                if mask is not None:
                    mask = mask.cuda(device)
            if args.multi:
                video = data[3].cuda()
                audio = data[4].cuda()
                log_prob = model(feat, lens, video, audio)
            else:
                log_prob = model(feat, lens)
            if isinstance(label, dict):
                loss1 = comput_class_loss(log_prob[0], label["ee"], weights[0])
                loss2 = comput_class_loss(log_prob[1], label["er"], weights[1])
                loss = 0.9*loss1 + 0.1*loss2
            else:
                loss = comput_class_loss(log_prob, label, weights, mask)
            if glob_steps > -1:
                loss.backward()
            report_loss += loss.item()
            glob_steps += 1

            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            model_opt.step()
            model_opt.zero_grad()

            if glob_steps % args.report_lr == 0:
                print("Steps: {}  LR: {}".format(glob_steps, model_opt.param_groups[0]['lr']))
                report_loss = 0

        # validate
        pAccs = emoeval(model=model, data_loader=dev_loader, args=args, weights=weights, dataname=dataname, key=key)
        print("Validate: ACCs-WA-UWA {}".format(pAccs))

        last_best = pAccs[-2]  # UWA
        # if args.dataset in ['IEMOCAP']:
        #     last_best = pAccs[-2] # WA
        if last_best > cur_best:
            if not os.path.isdir(args.save_dir):
                os.mkdir(args.save_dir)
            model_path = '{}/DGAT_{}.pt'.format(args.save_dir, args.dataset)
            torch.save(model.state_dict(), model_path)
            # Utils.model_saver(model, args.save_dir, args.type, args.dataset)
            cur_best = last_best
            over_fitting = 0
        else:
            over_fitting += 1
        print(over_fitting)

        if over_fitting >= args.patience:
            break
### Eval
def emoeval(model, data_loader, args, weights = None, dataname = "IEMOCAP", key = None):
    """ data_loader only input 'dev' """
    model.eval()
    
    val_loss = 0
    predidx = []
    gold = []
    predidx_pp = []
    gold_pp = []
    mask_all = []
    def append_pred(log_prob, label, predidx, gold):
        # accuracy
        emo_predidx = torch.argmax(log_prob, dim=1)
        emo_true = label.view(label.size(0))
        
        a = emo_predidx.cpu().tolist()
        for i in a:
            predidx.append(i)
        b = emo_true.cpu().tolist()
        for j in b:
            gold.append(j)
        return predidx, gold
    global_step = 0
    for data in tqdm(data_loader):
        global_step += 1
        feat = data[0]
        label = data[1]
        lens = data[2]
        if len(data) == 4:
            mask = data[3]
            mask_all = mask_all + mask.tolist()
        else:
            mask = None
        if args.gpu != None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            device = torch.device(0)
            model.cuda(device)
            feat = feat.cuda(device)
            if isinstance(label, dict):
                label = {k:v.cuda(device) for k,v in label.items()}
                if weights is not None:
                    weights = [w.cuda(device) for w in weights]
            else:
                label = label.cuda(device)
                if weights is not None:
                    weights = weights.cuda(device)
                if mask is not None:
                    mask = mask.cuda(device)

        if args.multi:
            video = data[3].cuda()
            audio = data[4].cuda()
            log_prob = model(feat, lens, video, audio)
        else:
            log_prob = model(feat, lens)
        if isinstance(label, dict):
            loss1 = comput_class_loss(log_prob[0], label["ee"], weights[0])
            loss2 = comput_class_loss(log_prob[1], label["er"], weights[1])
            loss = loss1 + loss2
        else:
            loss = comput_class_loss(log_prob, label, weights, mask)
        val_loss += loss.item()
        if isinstance(label, dict):
            predidx, gold = append_pred(log_prob[0], label["ee"], predidx, gold)
            predidx_pp, gold_pp = append_pred(log_prob[1], label["er"], predidx_pp, gold_pp)
        else:
            predidx, gold = append_pred(log_prob, label, predidx, gold)

    # f1 = metric(gold, predidx, dataset=dataname, classify=key)
    if isinstance(label, dict):
        f1_1 = metric(gold, predidx, dataset=dataname, classify=key)
        f1_2 = metric(gold_pp, predidx_pp, dataset=dataname, classify=key)
        f1 = f1_1# + f1_2# + [round((x+y)/2, 2) for x, y in zip(f1_1, f1_2)]
    else:
        f1 = metric(gold, predidx, dataset=dataname, classify=key, masks=mask_all if len(mask_all)>0 else None)
    # print(f1)
    Total = f1 + [val_loss / (global_step)]

    # Return to .train() state after validation
    model.train()

    return Total
### Loss
def comput_class_loss(log_prob, target, weights, mask = None):
    """ loss function """
    # if mask is not None:
    #     loss = F.nll_loss(log_prob * mask.unsqueeze(1), target.view(target.size(0)), weight=weights, reduction='sum')
    #     loss /= torch.sum(weights[target]*mask)
    # else:
    loss = F.nll_loss(log_prob, target.view(target.size(0)), weight=weights, reduction='sum')
    loss /= target.size(0)

    return loss


## Metric
def metric(labels, preds, dataset, classify = "emotion", masks=None):
    if dataset in ["IEMOCAP", 'MELD']:
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        fscores = [avg_fscore]

    elif dataset in ["Persuasion", "MultiWOZ"]:
        print("mask: {}".format(np.sum(np.array(masks))))
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
        fscores = [avg_fscore1, avg_fscore2, avg_fscore3]

    elif dataset == "DailyDialog":
        if classify == 'emo':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[1,2,3,4,5,6])*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore4 = round(f1_score(labels, preds, sample_weight=masks, average='micro', labels=[1,2,3,4,5,6])*100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            avg_fscore6 = round(f1_score(labels, preds, sample_weight=masks, average='macro', labels=[1,2,3,4,5,6])*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]

        elif classify == 'act':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
    return fscores