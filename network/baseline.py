# Implementation of Vision Transformer
# From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
import torch
from torch import nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
# helpers

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

class ViT(nn.Module):
    def __init__(self, *, input_dim=4096, mlp_dim=128, dim_head=64, output_channel=1, depth=5, heads=6, pool = 'reg', dropout = 0.1, emb_dropout = 0.1, max_length=1000):
        super().__init__()
        
        assert pool in {'reg', 'mean'}, 'pool type must be either reg (reg token) or mean (mean pooling)'
        #reduce the dimension of the input embeddings
        self.reduce_embedding = nn.Linear(input_dim, mlp_dim, bias=False)
        self.pos_embedding = PositionalEncoding(mlp_dim, max_len=max_length+1)
        # self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1, mlp_dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(mlp_dim, depth, heads, dim_head, mlp_dim*4, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, output_channel, bias=False)
        )
        self.heads = heads
        self.mlp_dim = mlp_dim

    def forward(self, x, video_len, max_len):
        x = self.reduce_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # pos_emb = [get_sinusoid_position_encoding(int(l.item()), self.mlp_dim) for l in (video_len+1)]
        
        x = self.pos_embedding(x)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        # pad_mask: [batch_size, n_heads, max_len, max_len]
        pad_mask = get_attn_pad_mask(video_len+1, max_len+1, self.heads, self.mlp_dim)
        x = x*pad_mask[1]
        x = self.transformer(x, pad_mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

def test_VQATransformer():
    # [batch_szie, max_len, d_feat] [2, 3, 2]
    X = torch.tensor([[[1, 1], [1, 0], [0, 0]], [[2, 1], [1, 1], [3, 0]]], dtype=torch.float32)
    # X = torch.randn(2,10,8).cuda()
    X = X.cuda()
    video_len = torch.zeros([2, 1]).cuda()
    video_len[0, 0] = 2
    video_len[1, 0] = 3
    vqa = ViT(input_dim=2, mlp_dim=2, dim_head=2, output_channel=1, depth=2, heads=2, pool = 'reg', dropout = 0.1, emb_dropout = 0.1, max_length=3).cuda()
    outputs = vqa(X, video_len, 3)
    print(outputs)
    print(outputs.shape)

if __name__ == '__main__':
    torch.manual_seed(1995)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.backcompat.broadcast_warning.enabled = True
    test_VQATransformer()
