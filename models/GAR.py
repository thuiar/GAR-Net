import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .fusion.fusionPlugin import fusionPlugin
class GRUencoder(nn.Module):
    def __init__(self, d_emb, d_out, num_layers):
        super(GRUencoder, self).__init__()
        # default encoder 2 layers
        self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
                          bidirectional=True, num_layers=num_layers)

    def forward(self, sent, sent_lens):
        """
        :param sent: torch tensor, batch_size x seq_len x d_rnn_in
        :param sent_lens: numpy tensor, batch_size x 1
        :return:
        """
        device = sent.device
        # seq_len x batch_size x d_rnn_in
        sent_embs = sent.transpose(0,1)

        # sort by length
        s_lens, idx_sort = np.sort(sent_lens)[::-1], np.argsort(-sent_lens)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda(device)
        # idx_sort = idx_sort.cuda(device)
        s_embs = sent_embs.index_select(1, Variable(idx_sort))

        # padding
        s_lens = s_lens.copy()
        sent_packed = pack_padded_sequence(s_embs, s_lens)
        sent_output = self.gru(sent_packed)[0]
        sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

        # unsort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
        # idx_unsort = idx_unsort.cuda(device)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # batch x seq_len x 2*d_out
        output = sent_output.transpose(0,1)

        return output
class DialogEncoder(nn.Module):
    def __init__(self, d_h1, d_h2, layers=2, drop=0.5):
        super(DialogEncoder, self).__init__()
        self.dialog_gru = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=False)
        d_h2 = d_h1 + d_h2 #+ sp_size
        self.fc = nn.Sequential(
            nn.Linear(d_h2, d_h1),
            nn.Tanh()
        )
        self.out_dim = d_h1
        d_h2 = d_h1
        self.dropout_mid = nn.Dropout(drop)
        self.gat = nn.ModuleList([GAT(d_h2) for _ in range(layers)])
        self.output2 = nn.Sequential(
            nn.Linear(d_h2, d_h2),
            nn.Tanh(),
        )
    def forward(self, x, sp=None):
        # self.dialog_gru
        s_context = self.dialog_gru(x.unsqueeze(1))[0]
        s_context = s_context.transpose(0,1).contiguous()
        out = torch.cat([s_context, x.unsqueeze(0)], dim=-1)
        out = self.fc(out)
        out = self.dropout_mid(out)
        for m in self.gat:
            out = m(out)
        return self.output2(out.squeeze(0))
class SentenceEncoder(nn.Module):
    def __init__(self, d_emb, d_out, num_layers, drop = 0.5):
        super(SentenceEncoder, self).__init__()
        self.gru = GRUencoder(d_emb, d_out, num_layers)
        self.cnn = nn.Conv1d(d_out*2, 1, kernel_size=3, stride=1, padding=1)
        fc_size = d_out*2 + d_emb
        self.dropout_in = nn.Dropout(drop)
        self.fc = nn.Linear(fc_size, d_out)
    def forward(self, x, x_lens):
        gru_x = self.gru(x, x_lens)
        # gru_x = torch.tanh(gru_x)
        g = self.cnn(gru_x.transpose(1, 2)).transpose(1, 2)
        gate_x = torch.tanh(gru_x) * F.sigmoid(g)
        combined = [x, gate_x]
        combined = torch.cat(combined, dim=-1)
        s_embed = self.fc(combined)
        s_embed = torch.tanh(s_embed)
        s_embed = torch.max(s_embed, dim=1)[0]
        s_embed = self.dropout_in(s_embed)
        return s_embed
class GAR(nn.Module):
    def __init__(self, args, embedding):
        super(GAR, self).__init__()
        self.d_h2 = args.d_h2
        # load word2vec
        self.embeddings = embedding
        self.roberta = args.roberta
        self.multi = args.multi
        if args.multi:
            self.fusion = fusionPlugin(args)
        self.sent_encoder = SentenceEncoder(args.d_word_vec, args.d_h1, num_layers=1, drop=args.drop)
        self.dialog_encoder = DialogEncoder(args.d_h1, args.d_h2, args.ll, drop=args.drop)
        self.num_classes = args.num_classes
        d_h2 = self.dialog_encoder.out_dim#-sp_size
        self.classifier = nn.Linear(d_h2, self.num_classes)

    def forward(self, sents, lens, video = None, audio =None):
        if len(sents.size()) < 2:
            sents = sents.unsqueeze(0)
        if self.roberta:
            w_embed = self.embeddings(sents, lens)
        else:
            w_embed = self.embeddings(sents)
        # sentence
        s_embed = self.sent_encoder(w_embed, lens)
        # dialogs
        if self.multi:
            s_embed = self.fusion(s_embed, video, audio)
        out = self.dialog_encoder(s_embed)

        # classifier
        out = self.classifier(out)
        pred_scores = F.log_softmax(out, dim=1)
        return pred_scores

class GAT(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wa = nn.Linear(hidden_size, hidden_size)
        self.wb = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.wc = nn.Linear(2*hidden_size, 1)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x-> b, c, d
        # x = x.unsqueeze(0)
        a = self.wa(x).unsqueeze(2).expand(-1, -1, x.size(1), -1)
        b = self.wb(x).unsqueeze(1).expand(-1, x.size(1), -1, -1)
        c = torch.cat([a, b], dim=3)
        c = self.wc(c).squeeze(3)
        c = F.leaky_relu(c)
        c = F.softmax(c, dim=2)
        out = torch.einsum('blc, bcd->bld', c, self.wv(x))
        out = torch.tanh(self.mlp(out))+x
        return out
