# Implementation of Vision Transformer
# From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
from re import X
import torch
from torch import nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F 
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
# position embedding
def get_sinusoid_position_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])   # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])   # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)   #(1,N,d)
# get the mask
def get_attn_pad_mask(video_len, max_len, n_heads, dim):
    """
    :param video_len: [batch_size]
    :param max_len: int
    :param n_heads: int
    :return pad_mask: [batch_size, n_heads, max_len, max_len]
    """
    batch_size = video_len.shape[0]
    global_pad_mask = torch.zeros([batch_size, max_len, max_len])
    local_pad_mask = torch.zeros([batch_size, max_len, dim])
    for i in range(batch_size):
        length = int(video_len[i].item())
        for j in range(length):
            global_pad_mask[i, j, :length] = 1
            local_pad_mask[i, j, :length] = 1

    # pad_mask: [batch_size, n_heads, max_len, max_len]
    global_pad_mask = global_pad_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
    device = video_len.device
    global_pad_mask = global_pad_mask.to(device)
    # local_pad_mask = local_pad_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
    local_pad_mask = local_pad_mask.to(device)
    return [global_pad_mask, local_pad_mask]
# classes
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
        return self.dropout(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, mask):
        return self.fn(self.norm(x), mask)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            # nn.Dropout(dropout)
        )
    def forward(self, x, mask):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            # nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
       
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # dots.masked_fill_(mask, -1e9)
        attn = self.attend(dots)
        attn = attn * mask[0]

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x, mask) + x
        return x

class Linear_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False, theta=0.5):

        super(Linear_cd, self).__init__() 
        self.cov1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.theta = theta

    def forward(self, x):
        out_normal = self.cov1d(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            [C_out, C_in, h] = self.cov1d.weight.shape
            kernel_diff = self.cov1d.weight.sum(2)
            kernel_diff = kernel_diff[:, :, None]
            out_diff = F.conv1d(input=x, weight=kernel_diff)

            return x - out_diff

class Embedding(nn.Module):
    def __init__(self, input_dim, red_dim, step, max_len):
        super().__init__()
        self.step = step
        self.reduce_embedding = nn.Linear(input_dim*2, red_dim, bias=False)
        self.fc_diff = Linear_cd(in_channels=max_len, out_channels=max_len, kernel_size=step)
        self.input_dim = input_dim

    def forward(self, x):
        b, n, d = x.shape

        x_ = self.fc_diff(x)
        x = torch.cat((x, x_), dim=2)
        x = self.reduce_embedding(x)
        return x

# def feature_difference(features):
#     diff_feature = features[:, 1:] - features[:, :-1]
#     last_feature = torch.zeros_like(features[:, -1], device=features.device, dtype=torch.float32)
    
#     return torch.cat([last_feature.unsqueeze(1), diff_feature], dim=1)
def feature_difference(features):
    b, n, c = features.shape
    mean = features[:,:, :c//2]
    std = features[:,:, c//2:]
    diff_mean = mean[:, 1:,:] - mean[:, :-1,:]
    last_mean = torch.zeros_like(diff_mean[:, -1], device=features.device, dtype=torch.float32)
    diff_mean = torch.cat([last_mean.unsqueeze(1), diff_mean], dim=1)
    diff_std = std[:, 1:,:] + std[:, :-1,:]
    last_std = torch.zeros_like(diff_std[:, -1], device=features.device, dtype=torch.float32)
    diff_std = torch.cat([last_std.unsqueeze(1), diff_std], dim=1)
    
    return torch.cat([diff_mean, diff_std], 2)

class ViT(nn.Module):
    def __init__(self, *, input_dim=4096, mlp_dim=128, dim_head=64, output_channel=1, depth=5, heads=6, pool = 'reg', dropout = 0.1, emb_dropout = 0.1, max_length=1000):
        super().__init__()
        
        assert pool in {'reg', 'mean'}, 'pool type must be either reg (reg token) or mean (mean pooling)'
        #reduce the dimension of the input embeddings
        self.reduce_embedding_0 = nn.Linear(input_dim//8, mlp_dim, bias=False)
        self.reduce_embedding_1 = nn.Linear(input_dim//4, mlp_dim, bias=False)
        self.reduce_embedding_2 = nn.Linear(input_dim//2, mlp_dim, bias=False)
        self.reduce_embedding_3 = nn.Linear(input_dim, mlp_dim, bias=False)
        # self.embedding = Embedding(input_dim, mlp_dim, step=21, max_len=max_length)
        
        self.pos_embedding = PositionalEncoding(mlp_dim, max_len=max_length+1)
        # self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1, mlp_dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(mlp_dim, depth, heads, dim_head, mlp_dim*4, dropout)
        # self.transformer_1 = Transformer(mlp_dim, depth, heads, dim_head, mlp_dim*4, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_0 = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, output_channel, bias=False)
        )
        self.mlp_head_1 = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, output_channel, bias=False)
        )
        self.mlp_head_2 = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, output_channel, bias=False)
        )
        self.mlp_head_3 = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, output_channel, bias=False)
        )
        self.heads = heads
        self.mlp_dim = mlp_dim
    
    def forward_once(self, x, reg_tokens, video_len, max_len):

        b, n, c = x.shape
        x = torch.cat((reg_tokens.unsqueeze(1), x), dim=1)
        # x = torch.cat((reg_tokens, x), dim=1)
        
        x = self.pos_embedding(x)

        pad_mask = get_attn_pad_mask(video_len+1, max_len+1, self.heads, self.mlp_dim)
        x = x*pad_mask[1]
        x = self.transformer(x, pad_mask)
        x_reg = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return x, x_reg
    
    def forward(self, video, video_len, max_len):
        # x = self.reduce_embedding(x)

        # x = self.embedding(x)

        x = video[0]
        x = feature_difference(x)
        # x = torch.cat((x, feature_difference(x)), dim=2)
        x = self.reduce_embedding_0(x)
        b, n, _ = x.shape
        reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
        x = torch.cat((reg_tokens, x), dim=1)
        # pos_emb = [get_sinusoid_position_encoding(int(l.item()), self.mlp_dim) for l in (video_len+1)]
        
        x = self.pos_embedding(x)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        # pad_mask: [batch_size, n_heads, max_len, max_len]
        pad_mask = get_attn_pad_mask(video_len+1, max_len+1, self.heads, self.mlp_dim)
        x = x*pad_mask[1]
        x = self.transformer(x, pad_mask)
        x0_reg = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x0 = self.to_latent(x0_reg)
        x0 = self.mlp_head_0(x0)

        x1 = video[1]
        x1 = feature_difference(x1)
        # x1 = torch.cat((x1, feature_difference(x1)), dim=2)
        x1 = self.reduce_embedding_1(x1)
        # b, n, _ = x1.shape
        # reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
        x1, x1_reg = self.forward_once(x1, x0_reg, video_len, max_len)
        x1 = self.to_latent(x1_reg)
        x1 = self.mlp_head_1(x1)

        x2 = video[2]
        x2 = feature_difference(x2)
        # x2 = torch.cat((x2, feature_difference(x2)), dim=2)
        x2 = self.reduce_embedding_2(x2)
        # b, n, _ = x2.shape
        # reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
        x2, x2_reg = self.forward_once(x2, x1_reg, video_len, max_len)
        x2 = self.to_latent(x2_reg)
        x2 = self.mlp_head_2(x2)

        x3 = video[3]
        x3 = feature_difference(x3)
        # x3 = torch.cat((x3, feature_difference(x3)), dim=2)
        x3 = self.reduce_embedding_3(x3)
        # b, n, _ = x3.shape
        # reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
        x3, x3_reg = self.forward_once(x3, x2_reg, video_len, max_len)
        x3 = self.to_latent(x3_reg)
        x3 = self.mlp_head_3(x3)
        

        return x0, x1, x2, x3

def test_VQATransformer():

    X = [torch.randn(2,10,8).cuda(), torch.randn(2,10,16).cuda(), torch.randn(2,10,32).cuda(), torch.randn(2,10,64).cuda()]
    # X = torch.tensor([[[1, 1], [1, 0], [0, 0]], [[2, 1], [1, 1], [3, 0]]], dtype=torch.float32)
    video_len = torch.zeros([2, 1]).cuda()
    video_len[0, 0] = 5
    video_len[1, 0] = 8
    vqa = ViT(input_dim=64, mlp_dim=4, dim_head=2, output_channel=1, depth=2, heads=2, pool = 'reg', dropout = 0.1, emb_dropout = 0.1, max_length=10).cuda()
    outputs = vqa(X, video_len, 10)
    print(outputs)
    print(outputs.shape)

if __name__ == '__main__':
    torch.manual_seed(1995)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.backcompat.broadcast_warning.enabled = True
    test_VQATransformer()
