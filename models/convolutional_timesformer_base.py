import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from models.efficientnet.efficientnet_pytorch import EfficientNet


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
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
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

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v, mask = cls_mask)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))
       
        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_, mask = mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

# main classes

class ConvolutionalTimeSformer(nn.Module):
    def __init__(
        self,
        *,
        config
    ):
        
        super().__init__()
        self.dim = config['model']['dim']
        self.num_frames = config['model']['num-frames']
        self.num_patches = config['model']['num-patches']
        self.image_size = config['model']['image-size']
        self.num_classes = config['model']['num-classes']
        self.patch_size = config['model']['patch-size']
        self.channels = config['model']['channels']
        self.depth = config['model']['depth']
        self.heads = config['model']['heads']
        self.dim_head = config['model']['dim-head']
        self.attn_dropout = config['model']['attn-dropout']
        self.ff_dropout = config['model']['ff-dropout']
        self.shift_tokens = config['model']['shift-tokens']
        self.efficient_net_block = config['model']['efficient-net-block']
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        

        for m in self.efficient_net.modules():
            m.requires_grad = False
        self.efficient_net.eval()
        
        
        num_positions = self.num_frames * self.num_patches
        patch_dim = self.patch_size ** 2

 

        self.to_patch_embedding = nn.Linear(patch_dim, self.dim)
        self.cls_token = nn.Parameter(torch.randn(1, self.dim))
     
        self.pos_emb = nn.Embedding(num_positions + 1, self.dim)
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

    def forward(self, x, mask = None, size_embedding = None):
        b, f, h, w, _, *_, device, p = *x.shape, x.device, self.patch_size
        hp, wp = (h // p), (w // p)
        n = hp * wp
        
        x = rearrange(x, 'b f h w c -> (b f) c h w') 
        x = self.efficient_net.extract_features_at_block(x, self.efficient_net_block)
        x = rearrange(x, '(b f) c h w -> b f c h w', b = b, f = f) 
        x = rearrange(x, 'b f c h w -> b (f c) (h w)')
        tokens = self.to_patch_embedding(x)
        
        # add cls token
        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x =  torch.cat((cls_token, tokens), dim = 1)
        # positional embedding
        x += self.pos_emb(torch.arange(x.shape[1], device = device))
        
        # size embedding
        size_embedding = repeat(size_embedding, 'b f -> p b f', p=self.num_patches) 
        size_embedding = rearrange(size_embedding, 'p b f -> (p b f)')
        size_embedding = torch.cat((torch.tensor([0]), size_embedding), dim = 0)
        size_embedding = size_embedding.to(device)
        x += self.size_emb(size_embedding)

        # calculate masking for uneven number of frames

        frame_mask = None
        cls_attn_mask = None

        # time and space attention

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        return self.to_out(cls_token)
