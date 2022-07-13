"""Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.GRU_v0 import GRUModel

class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers - 1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VSFA(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32):
        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        # self.ann_1 = ANN(reduced_size, hidden_size, 1) 
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        # self.rnn = GRUModel(reduced_size, hidden_size, 1)
        
        self.q = nn.Linear(hidden_size, 1)
        self.s = nn.Linear(1250, 1)

    def forward(self, input, input_length):
        # self.rnn.flatten_parameters()
        b, n, *_ = input.shape
        # print('input shape', input.shape)
        input = self.ann(input)  # dimension reduction
        # outputs = self.ann_1(input)
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        # outputs, _ = self.rnn(input)
        q = self.q(outputs)  # frame quality
        score = self.s(q.view(b, -1, n)).squeeze()
        score = torch.mean(q, 1)
        # score = torch.zeros(input_length, device=q.device)
        # score = torch.zeros_like(input_length, device=q.device, dtype=torch.float32)
        # for i in range(input_length.shape[0]):
        #     qi = q[i, :np.int(input_length[i].cpu().numpy())] #[n, c]
        #     qi = TP(qi)
        #     score[i] = torch.mean(qi)  # video overall quality
        return score.squeeze()

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


def feature_difference(features):
    diff_feature = features[:, 1:] - features[:, :-1]
    last_feature = torch.zeros_like(features[:, -1], device=features.device, dtype=torch.float32)
    diff_feature.append(last_feature)
    return diff_feature
