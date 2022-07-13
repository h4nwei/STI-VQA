import torch
from torch import nn, einsum
from torch._C import device
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x, mask, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = attn * mask
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_(nn.Module):
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

    def forward(self, x):
       
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # dots.masked_fill_(mask, -1e9)
        attn = self.attend(dots)
        attn = attn 

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask):
        for norm_attn, attn, norm_ffd, ff in self.layers:
            x = norm_attn(attn(x, mask)) + x
            x = norm_ffd(ff(x)) + x
        return x

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        tokens = torch.cat((sm_tokens, lg_tokens), dim = 1)
        for attend, ff in self.layers:
            tokens = attend(tokens) + tokens
            tokens = ff(tokens) + tokens
        tokens = tokens.mean(dim = 1) 

        return tokens

# multi-scale encoder

class MultiRangeEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        long_dim,
        short_dim,
        cross_mlp_dim,
        long_enc_params,
        short_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = long_dim, dropout = dropout, **long_enc_params),
                Transformer(dim = short_dim, dropout = dropout, **short_enc_params),
                CrossTransformer(dim=long_dim, depth=cross_attn_depth, heads=cross_attn_heads, 
                mlp_dim=cross_mlp_dim, dim_head=cross_attn_dim_head, dropout=dropout)
            ]))

    def forward(self, long_tokens, short_tokens, long_pad_mask, short_pad_mask):
        for long_enc, short_enc, cross_attend in self.layers:
            long_tokens = long_enc(long_tokens, long_pad_mask)
            short_tokens = [short_enc(short_tokens[:,i,:,:], short_pad_mask) for i in range(short_tokens.shape[1])]
            short_tokens = torch.stack(short_tokens, 1)
            long_reg_tokens = long_tokens[:,:1]
            short_reg_token = short_tokens[:,:,:1].squeeze(2)
            tokens = cross_attend(long_reg_tokens, short_reg_token)

        return tokens

# patch-based image to token embedder

class Embedding(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        red_dim,
        step=1,
        max_len,
        status='global',
        dropout = 0.1
    ):
        super().__init__()
        self.status = status
        self.step = step
        self.reduce_embedding = nn.Linear(input_dim, red_dim, bias=False)
        self.pos_embedding_global = nn.Parameter(torch.randn(1, max_len + 1, red_dim))
        self.pos_embedding_local = nn.Parameter(torch.randn(1, self.step*2 + 1, red_dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, red_dim))
        self.pad_tokens = nn.Parameter(torch.zeros(1, self.step//2, input_dim))
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.input_dim = input_dim

    def forward(self, x, mask):
        b, n, d = x.shape
        if self.status == 'global':
            x = self.reduce_embedding(x)
            reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)
            x = torch.cat((reg_tokens, x), dim=1)
            x += self.pos_embedding_global[:, :(n + 1)]
            x = x * mask
        else:
            interval = self.step // 2
            pad_tokens = repeat(self.pad_tokens, '() n d -> b n d', b = b)
            x = torch.cat((pad_tokens, x, pad_tokens), dim=1)
            reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b).to(x.device)
            x_ = torch.zeros(b, n, self.step*2 + 1, reg_tokens.shape[-1]).to(x.device)
            for i in range(interval, x.shape[1]-interval*2):
                # print(i)
                tmp = x[:, i-interval:i+interval+1, :] 
                x_diff = x[:, i-interval:i+interval+1, :] - tmp[:, interval, :].unsqueeze(1)
                x_new = torch.cat((x[:, i-interval:i+interval+1, :], x_diff), dim=1)
                x_new_red = self.reduce_embedding(x_new)
                x_[:,i,:,:] = (torch.cat((reg_tokens, x_new_red), dim=1) + self.pos_embedding_local)*mask
            x = x_
            
        return self.dropout(x)


class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        num_output,
        depth,
        global_dim,
        global_enc_depth,
        global_enc_heads,
        global_enc_mlp_dim,
        local_dim,
        local_enc_depth,
        local_enc_heads,
        local_enc_mlp_dim,
        local_embedding_len,
        cross_mlp_dim, 
        max_len,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.long_range_embedder = Embedding(input_dim=input_dim, red_dim=global_dim, max_len=max_len, step=1, dropout=emb_dropout, status='global')
        self.short_range_embedder = Embedding(input_dim=input_dim, red_dim=local_dim, max_len=max_len, step=local_embedding_len, dropout=emb_dropout, status='local')

        self.multi_range_encoder = MultiRangeEncoder(
            depth = depth,
            long_dim = global_dim,
            short_dim = local_dim,
            cross_mlp_dim=cross_mlp_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            long_enc_params = dict(
                depth = global_enc_depth,
                heads = global_enc_heads,
                mlp_dim = global_enc_mlp_dim,
                dim_head = 64
            ),
            short_enc_params = dict(
                depth = local_enc_depth,
                heads = local_enc_heads,
                mlp_dim = local_enc_mlp_dim,
                dim_head = 64
            ),
            dropout = dropout
        )
        self.global_enc_mlp_dim = global_enc_mlp_dim
        self.local_enc_mlp_dim = local_enc_mlp_dim
        self.global_enc_heads = global_enc_heads
        self.local_enc_heads = local_enc_heads
        self.local_embedding_len = local_embedding_len
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(local_dim, 1, bias=False)
        )

    def forward(self, x, video_len, max_len):
        b, n, d = x.shape
        local_len = torch.ones(b, 1)*self.local_embedding_len
        local_len = local_len.to(x.device)
        long_pad_mask = get_attn_pad_mask(video_len+1, max_len+1, self.global_enc_heads, self.global_dim)
        short_pad_mask = get_attn_pad_mask(local_len+1, self.local_embedding_len*2+1, self.local_enc_heads, self.local_dim)
    
        long_tokens = self.long_range_embedder(x, long_pad_mask[1])
        short_tokens = self.short_range_embedder(x, short_pad_mask[1])
        result = self.multi_range_encoder(long_tokens, short_tokens, long_pad_mask[0], short_pad_mask[0])
        result = self.mlp_head(result)

        return result

if __name__ == '__main__':

    v = CrossViT(
        input_dim = 8,        # dimension of the input embedding 
        num_output = 1,          # dimension of the output embedding
        depth = 4,               # number of multi-scale encoding blocks
        global_dim = 4,            # long-range dependence dimension
        global_enc_depth = 2,        # long-range dependence depth
        global_enc_heads = 4,        # long-range dependence heads
        global_enc_mlp_dim = 4,   # long-range dependence feedforward dimension
        local_dim = 4,            # short-range dependence dimension
        local_enc_depth = 2,        # short-range dependence depth
        local_enc_heads = 4,        # short-range dependence heads
        local_embedding_len = 3, # length of the local range
        local_enc_mlp_dim = 4,   # short-range dependence feedforward dimensions
        cross_mlp_dim = 4,
        cross_attn_depth = 2,    # cross attention depths
        cross_attn_heads = 4,    # cross attention heads
        max_len = 10,
        dropout = 0.1,
        emb_dropout = 0.1
    ).cuda()
    X = torch.randn(2,10,8).cuda()
    # X = torch.tensor([[[1, 1], [1, 0], [0, 0]], [[2, 1], [1, 1], [3, 0]]], dtype=torch.float32)
    video_len = torch.zeros([2, 1]).cuda()
    video_len[0, 0] = 5
    video_len[1, 0] = 8

    # img = torch.randn(2, 400, 256)

    pred = v(X, video_len, 10) # (1, 1000)
    print(pred)

