import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.GRU import dGRUModel

class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


def feature_difference(features):
    diff_feature = features[:, 1:] - features[:, :-1]
    last_feature = torch.zeros_like(features[:, -1], device=features.device, dtype=torch.float32)
    
    return torch.cat([last_feature.unsqueeze(1), diff_feature], dim=1)






class VSFA_new(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32):

        super(VSFA_new, self).__init__()
        self.hidden_size = hidden_size
        self.ann_s = ANN(input_size, reduced_size, 1)
        self.ann_t = ANN(input_size, reduced_size, 1)
        self.rnn_feature = dGRUModel(reduced_size, hidden_size, layer_dim=1)


    def forward(self, input, len):
        s, t = torch.split(input, [4096,4096], 2)
        input_s = self.ann_s(s)  # dimension reduction
        input_t = self.ann_s(t)  # dimension reduction
        
        score = self.rnn_feature(input_s, input_t)

        return score
        
        
        # return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0