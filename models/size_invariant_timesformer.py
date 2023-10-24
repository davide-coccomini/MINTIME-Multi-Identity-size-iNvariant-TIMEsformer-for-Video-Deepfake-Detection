import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from statistics import mean
from models.efficientnet.efficientnet_pytorch import EfficientNet
from torch.nn.init import trunc_normal_
import cv2
import numpy as np
from random import random


# helpers
def exists(val):
    return val is not None

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt is 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f = f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim = -1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim = -1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim = 1)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out, attn

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, identities_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out, cls_attentions = attn(cls_q, k, v, mask = cls_mask)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))
       
        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out, attentions = attn(q_, k_, v_, mask = mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)
      
        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        
        # combine heads out
        return self.to_out(out), cls_attentions


class SizeInvariantTimeSformer(nn.Module):
    def __init__(
        self,
        *,
        config,
        require_attention = False
    ):
        
        super().__init__()
        self.dim = config['model']['dim']
        self.num_frames = config['model']['num-frames']
        self.max_identities = config['model']['max-identities']
        self.image_size = config['model']['image-size']
        self.num_classes = config['model']['num-classes']
        self.patch_size = config['model']['patch-size']
        self.num_patches = config['model']['num-patches']
        self.channels = config['model']['channels']
        self.depth = config['model']['depth']
        self.heads = config['model']['heads']
        self.dim_head = config['model']['dim-head']
        self.attn_dropout = config['model']['attn-dropout']
        self.ff_dropout = config['model']['ff-dropout']
        self.shift_tokens = config['model']['shift-tokens']
        self.enable_size_emb = config['model']['enable-size-emb']
        self.enable_pos_emb = config['model']['enable-pos-emb']
        self.require_attention = require_attention

        num_positions = self.num_frames * self.channels
        self.to_patch_embedding = nn.Linear(self.channels , self.dim)
        self.cls_token = nn.Parameter(torch.randn(1, self.dim))
     
        self.pos_emb = nn.Embedding(num_positions + 1, self.dim)
        if self.enable_size_emb:
            self.size_emb = nn.Embedding(num_positions + 1, self.dim)


        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            ff = FeedForward(self.dim, dropout = self.ff_dropout)
            time_attn = Attention(self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout)
            spatial_attn = Attention(self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout)
            if self.shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))

           
            time_attn, spatial_attn, ff = map(lambda t: PreNorm(self.dim, t), (time_attn, spatial_attn, ff))
            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        # Initialization
        trunc_normal_(self.pos_emb.weight, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.enable_size_emb:
            trunc_normal_(self.size_emb.weight, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.enable_size_emb:
            return {'pos_emb', 'cls_token', 'size_emb'}
        else:
            return {'pos_emb', 'cls_token'}


    def forward(self, x, mask = None,  identities_mask = None, size_embedding = None, positions = None):
        b, f, c, h, w, *_, device = *x.shape, x.device
        n = h * w
        x = rearrange(x, 'b f c h w -> b (f h w) c')                                   # B x F*P*P x C
        tokens = self.to_patch_embedding(x)                                            # B x 8*7*7 x dim

        # Add cls token
        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x = torch.cat((cls_token, tokens), dim = 1)

        # Positional 
        if self.enable_pos_emb:
            x += self.pos_emb(positions)
        else:
            x += (self.pos_emb(torch.arange(x.shape[1]).to(device)))

        # Size embedding
        if self.enable_size_emb:
            size_embedding = repeat(size_embedding, 'b f -> b f p', p=self.num_patches)      # B x 8 x 49   
            size_embedding = rearrange(size_embedding, 'b f p -> b (f p)')
            cls_token = torch.Tensor([0]*b).unsqueeze(-1).to(device)
            size_embedding = size_embedding.to(device)
            size_embedding = torch.cat((cls_token, size_embedding), dim = 1)
            size_embedding = size_embedding.to(device).int()
            x += self.size_emb(size_embedding)
       
                
        # Frame mask            
        frame_mask = repeat(mask, 'b f1 -> b f2 f1', f2 = self.num_frames)
        frame_mask = torch.logical_and(frame_mask, identities_mask)
        frame_mask = F.pad(frame_mask, (1, 0), value= True)
        frame_mask = repeat(frame_mask, 'b f1 f2 -> (b h n) f1 f2', n = n, h = self.heads)
      

        # CLS mask
        cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n = n, h = self.heads)
        cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value = True)

        # Time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            y, time_attention = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask)
            x = x + y
            y, space_attention = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask)
            x = x + y
            x = ff(x) + x

        cls_token = x[:, 0]
        attentions = [space_attention, time_attention]

        if self.require_attention:
            return self.to_out(cls_token), attentions
        else:
            return self.to_out(cls_token)
