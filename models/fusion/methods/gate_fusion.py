import torch
import torch.nn as nn
import torch.nn.functional as F

class GateFusion(nn.Module):
    def __init__(self, args):
        super(GateFusion, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.video_in, self.audio_in = args.fusion_input_features
        # write utterance_dim into args. 
        args.fusion_utterance_dim = self.text_in
        self.fc1 = nn.Linear(self.text_in, self.text_in)
        self.fc2 = nn.Linear(self.video_in, self.text_in)
        self.fc3 = nn.Linear(self.audio_in, self.text_in)
        self.fc4 = nn.Linear(2 * self.text_in, self.text_in)
        self.cnn = nn.Conv1d(self.text_in, )
        self.gate = nn.Linear(2 * self.text_in, 1)
        self.drop1 = nn.Dropout(args.drop)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, text_x, video_x, audio_x):
        '''
        Args:
            audio_x: tensor of shape (seq_len, batch_size, audio_in)
            video_x: tensor of shape (seq_len, batch_size, video_in)
            text_x: tensor of shape  (seq_len, batch_size, text_in)
        '''
        ori_x = text_x
        text_x = torch.tanh(self.fc1(text_x))
        fusion_ = torch.cat([self.fc2(video_x), self.fc3(audio_x)], -1)
        fusion_ = torch.tanh(self.fc4(fusion_))
        fusion_ = self.drop1(fusion_)
        gate_in = torch.cat([text_x, fusion_], -1)
        g = self.gate(gate_in).sigmoid()
        out = g * text_x + (1 -g)*fusion_
        out = self.drop2(out)

        return out  + ori_x