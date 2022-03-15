"""
paper: Tensor Fusion Network for Multimodal Sentiment Analysis
From: https://github.com/A2Zadeh/TensorFusionNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

class TFN_S(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''
    def __init__(self, args):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN_S, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.text_in, self.video_in, self.audio_in = args.fusion_input_features
        self.text_hidden, _ , self.audio_video_hidden = args.pre_fusion_hidden_dims
        self.pre_fusion_dropout = args.pre_fusion_dropout
        self.post_fusion_dropout = args.post_fusion_dropout
        self.post_fusion_dim = self.text_in # use text_in dim for the utterance representation dim. & use residents.

        # write utterance_dim into args.
        args.fusion_utterance_dim = self.post_fusion_dim

        # define the pre-fusion subnetworks
        self.trans_audio_video = nn.Sequential()
        self.trans_audio_video.add_module('audio_video_norm', nn.BatchNorm1d(self.audio_in + self.video_in))
        self.trans_audio_video.add_module('audio_video_linear', nn.Linear(in_features=self.audio_in + self.video_in , out_features=self.audio_video_hidden))
        self.trans_audio_video.add_module('audio_video_dropout', nn.Dropout(self.pre_fusion_dropout))
        self.trans_text = nn.Sequential()
        self.trans_text.add_module('text_norm', nn.BatchNorm1d(self.text_in))
        self.trans_text.add_module('text_linear', nn.Linear(in_features=self.text_in, out_features=self.text_hidden))
        self.trans_text.add_module('text_dropout', nn.Dropout(self.pre_fusion_dropout))

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.audio_video_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)



    def forward(self, text_x, video_x, audio_x):
        '''
        Args:
            audio_x: tensor of shape (seq_len, batch_size, audio_in)
            video_x: tensor of shape (seq_len, batch_size, video_in)
            text_x: tensor of shape  (seq_len, batch_size, text_in )
        '''
        seq_len    = text_x.shape[0]
        batch_size = text_x.shape[1]

        audio_video_x = torch.cat((audio_x, video_x), dim=2) # (seq_len, batch_size, audio_in + video_in)
        audio_video_h = self.trans_audio_video(audio_video_x.view(seq_len * batch_size, self.audio_in + self.video_in)) # seq_len, batch_size, audio_hidden

        text_h  = self.trans_text( text_x.view( seq_len * batch_size, self.text_in)) # seq_len, batch_size,  text_hidden

        # out production.
        constant_one = torch.ones(size=[batch_size * seq_len, 1], requires_grad=False).type_as(text_x).to(text_x.device)

        # _audio_h = torch.cat((constant_one, audio_h), dim=1)
        # _video_h = torch.cat((constant_one, video_h), dim=1)
        _audio_video_h = torch.cat((constant_one, audio_video_h), dim=1)
        _text_h  = torch.cat((constant_one,  text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_video_h.unsqueeze(2), _text_h.unsqueeze(1)).view(seq_len, -1) # (batch_size, audio_in + 1, video_in + 1)
        # print(fusion_tensor.shape)
        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        # print(post_fusion_dropped.shape)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        # print(post_fusion_y_1.shape)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True).unsqueenze(1)
        # print(post_fusion_y_2.shape)
        # exit()

        return post_fusion_y_2 + text_x # residents
