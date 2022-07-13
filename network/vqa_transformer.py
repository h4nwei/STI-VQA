import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# from network.vsfa_new import feature_difference



def get_attn_pad_mask(video_len, max_len, n_heads):
    """
    :param video_len: [batch_size]
    :param max_len: int
    :param n_heads: int
    :return pad_mask: [batch_size, n_heads, max_len, max_len]
    """
    batch_size = video_len.shape[0]
    pad_mask = torch.zeros([batch_size, max_len, max_len])

    for i in range(batch_size):
        length = int(video_len[i].item())
        for j in range(length):
            pad_mask[i, j, :length] = 1

    # pad_mask: [batch_size, n_heads, max_len, max_len
    pad_mask = pad_mask.eq(0).unsqueeze(1).repeat(1, n_heads, 1, 1)
    device = video_len.device
    pad_mask = pad_mask.to(device)
    return pad_mask

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        '''
        :param Q: [batch_size, n_heads, max_len, d_q]
        :param K: [batch_size, n_heads, max_len, d_k]
        :param V: [batch_size, n_heads, max_len, d_v]
        :param attn_mask: [batch_size, n_heads, max_len, max_len]
        :return:
        '''
        d_q = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_q)
        scores.masked_fill_(attn_mask, -1e9)

        attn = self.softmax(scores)
        Z = torch.matmul(attn, V)
        return Z


class EncoderLayer(nn.Module):
    def __init__(self, d_feat, d_q, d_v, n_heads, init_weights=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.n_heads = n_heads
        d_feat = d_feat
        self.d_q = d_q # d_q == d_k
        self.d_v = d_v

        # Attention
        self.Wq = nn.Linear(d_feat, d_q * n_heads, bias=False)
        self.Wk = nn.Linear(d_feat, d_q * n_heads, bias=False)
        self.Wv = nn.Linear(d_feat, d_v * n_heads, bias=False)
        self.attention = Attention()
        self.Wo =  nn.Sequential(nn.Linear(
            n_heads * d_v, d_feat, bias=False),
            nn.Dropout(dropout)
        )
        self.attn_layernorm = nn.LayerNorm(d_feat)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_feat, d_feat*4, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feat*4, d_feat, bias=False),
            nn.Dropout(dropout),
        )
        self.feed_layernorm = nn.LayerNorm(d_feat)

        # 初始化参数
        if init_weights:
            self.apply(init_weights)

    def forward(self, X, pad_mask):
        """
        :param X: [batch_size, max_len, d_feat]
        :param pad_mask: [batch_size, n_heads, max_len, max_len]
        :return:
        """
        # Attention
        batch_size, _, _ = X.shape
        # [batch_size, n_heads, max_len, d_q]
        Q = self.Wq(X).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)
        # [batch_size, n_heads, max_len, d_q], d_q == d_k
        K = self.Wk(X).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)
        # [batch_size, n_heads, max_len, d_v]
        V = self.Wv(X).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # [batch_size, n_heads, max_len, d_v]
        Z = self.attention(Q, K, V, pad_mask)
        # [batch_size, max_len, n_heads*d_v]
        Z = Z.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.d_v)
        # [batch_size, max_len, d_feat]
        Z = self.Wo(Z)
        Z = self.attn_layernorm(Z+X) # residual

        # Feed forward
        Z = self.feed_forward(Z)
        Z = self.feed_layernorm(Z + X) # residual

        return Z


class VQATransformer(nn.Module):
    def __init__(self, n_layers, d_feat, d_red, d_q, d_v, n_heads, max_len=1000, dropout=0.1, emb_dropout=0.1, use_pos_enc=False, init_weights=None, d_output=1):
        """
        :param n_layers:
        :param d_feat:
        :param d_red: 将输入数据d_feat维度减少到d_red维度
        :param d_q:
        :param d_v:
        :param n_heads:
        :param d_output: 用于使用zwx论文中通过IQA预训练模型中需要输出有两种的情况
        """
        super(VQATransformer, self).__init__()
        self.n_heads = n_heads
        # 加在每一个视频的第一帧前，用于最终分数的预测
        self.video_score_token = nn.Parameter(torch.zeros(1, 1, d_feat))
        self.dim_red = nn.Linear(d_feat, d_red, bias=False)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_red, d_q, d_v, n_heads, init_weights=init_weights, dropout=dropout) for _ in range(n_layers)])
        # 用于获得最后的分数
        self.feat2score = nn.Linear(d_red, d_output)
        self.dropout = nn.Dropout(emb_dropout)
        # 位置编码
        self.use_pos_enc = use_pos_enc
        self.pos_enc = PositionalEncoding(d_red, max_len=max_len+1)
        self.register_parameter('pos_embedding', nn.Parameter(torch.randn(1, max_len+1, d_red)))
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
        # if init_weights:
        #     self.apply(init_weights)
    def feature_difference(self, features):
        diff_feature = features[:, :, 1:] - features[:, :, :-1]
        last_feature = torch.zeros_like(features[:,:, -1], device=features.device, dtype=torch.float32)
    
        return torch.cat([last_feature.unsqueeze(2), diff_feature], dim=2)
    def forward(self, X, video_len, max_len):
        """
        :param X: [batch_size, max_len, d_feat]
        :param video_len: [batch_size, 1]
        :param max_len: int, video max length
        :return:
        """
        # rearrange X
        # X = torch.cat([self.feature_difference(X[:, :, 4096:8192]), X[:, :, 4096:8192]], 2)
        # [1, 1, d_feat] -> [B, 1, d_feat]
        video_score_token = self.video_score_token.expand(X.shape[0], -1, -1)
        # X: [batch_size, max_len, d_feat] -> [batch_size, max_len+1, d_feat]
        X = torch.cat((video_score_token, X), dim=1)
        # [batch_size, max_len+1, d_feat] -> [batch_size, max_len+1, d_red]
        X = self.dim_red(X)
        b, n, _ = X.shape
        # 位置编码
        if self.use_pos_enc:
            X += self.pos_embedding[:, :(n)]
            # X = self.pos_enc(X)
        X = self.dropout(X)
        # pad_mask: [batch_size, n_heads, max_len, max_len]
        pad_mask = get_attn_pad_mask(video_len+1, max_len+1, self.n_heads)
        for layer in self.encoder_layers:
            X = layer(X, pad_mask)
        # 获取添加的第0帧用于映射得到分数
        # [batch_size, d_red]
        X = X[:, 0]
        # [batch_sie, 1]
        scores = self.feat2score(X)
        
        return scores

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x.permute(1, 0, 2) # [batch_size, max_len, d_model] --> [max_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1, 0, 2)
        return x#self.dropout(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')

def test_Attention():
    X = torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.float32)
    W_q = torch.tensor([[1,1,0], [0,1,1]], dtype=torch.float32)
    W_k = torch.tensor([[1,1,0], [1,0,1]], dtype=torch.float32)
    W_v = torch.tensor([[1,1,0,1], [1,0,1,1]], dtype=torch.float32)

    Q = torch.matmul(X, W_q).unsqueeze(dim=1)
    K = torch.matmul(X, W_k).unsqueeze(dim=1)
    V = torch.matmul(X, W_v).unsqueeze(dim=1)
    video_len = torch.zeros([1, 1])
    video_len[0, 0] = 2
    pad_mask = get_attn_pad_mask(video_len, 3, 1)
    attn = Attention()
    Z = attn(Q, K, V, pad_mask)
    print(Z)
    print(Z.shape)

def test_EncoderLayer():
    X = torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.float32)
    video_len = torch.zeros([1, 1])
    video_len[0, 0] = 2
    pad_mask = get_attn_pad_mask(video_len, 3, 1)
    encoderlayer = EncoderLayer(2, 3, 4, 1)
    Z = encoderlayer(X, pad_mask)
    print(Z)
    print(Z.shape)
    print(X.shape)



def test_VQATransformer():
    # [batch_szie, max_len, d_feat] [2, 3, 2]
    X = torch.tensor([[[1, 1], [1, 0], [0, 0]], [[2, 1], [1, 1], [3, 0]]], dtype=torch.float32)
    # X = torch.randn(2,10,8).cuda()
    X = X.cuda()
    video_len = torch.zeros([2, 1]).cuda()
    video_len[0, 0] = 2
    video_len[1, 0] = 3
    vqa = VQATransformer(2, 2, 2, 2, 2, 2, max_len=10, dropout=0.1, emb_dropout=0.1, use_pos_enc=True, init_weights=None, d_output=1).cuda()
    outputs = vqa(X, video_len, 3)
    print(outputs)
    print(outputs.shape)

if __name__ == '__main__':
    torch.manual_seed(1995)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(20210810)
    torch.utils.backcompat.broadcast_warning.enabled = True
    # vqa = VQATransformer(3, 2, 1, 3, 4, 1, use_pos_enc=True, init_weights=init_weights)
    test_VQATransformer()

