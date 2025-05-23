import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SparseSoftHyperedgeGeneration(nn.Module):
    def __init__(self, node_dim, num_dyn_hyperedges, k, num_fixed_hyperedges=0,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_dyn_hyperedges  
        self.k = k
        self.num_fixed_hyperedges = num_fixed_hyperedges
        self.total_hyperedges = num_fixed_hyperedges + num_dyn_hyperedges  
        self.head_dim = node_dim // num_heads

        self.prototype_base = nn.Parameter(torch.Tensor(self.total_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)

        self.context_net = nn.Linear(2 * node_dim, self.total_hyperedges * node_dim)
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape

        avg_context = X.mean(dim=1)           
        max_context, _ = X.max(dim=1)         
        context_cat = torch.cat([avg_context, max_context], dim=-1) 

        prototype_offsets = self.context_net(context_cat).view(B, self.total_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets  

        X_proj = self.pre_head_proj(X)  
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.total_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.total_hyperedges, self.head_dim).transpose(1, 2)
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.total_hyperedges).mean(dim=1)
        logits = self.dropout(logits)
        
        A = logits  
        
        global_scores = A.sum(dim=1)  
        dynamic_scores = global_scores[:, self.num_fixed_hyperedges:] 
        topk_vals, topk_indices = torch.topk(dynamic_scores, self.k, dim=-1)  
        topk_indices += self.num_fixed_hyperedges

        mask = torch.zeros_like(global_scores)  
        mask[:, :self.num_fixed_hyperedges] = 1.0  
        mask.scatter_(dim=1, index=topk_indices, src=torch.ones_like(topk_vals))
        mask = mask.unsqueeze(1) 

        A = A * mask
        A = F.softmax(A, dim=1)

        self.last_mask = mask.detach()
        self.last_A = A.detach()
        return A

    def load_balance_loss(self):
        B = self.last_mask.shape[0]
        dynamic_mask = self.last_mask[:, :, self.num_fixed_hyperedges:]  
        global_activation = dynamic_mask.squeeze(1).mean(dim=0) 
        target = self.k / self.num_hyperedges
        return ((global_activation - target) ** 2).mean()

class SoftHGNN_SeS(nn.Module):
    def __init__(self, embed_dim, num_dyn_hyperedges=64, top_k=16, num_fixed_hyperedges=8,
                 num_heads=8, dropout=0.1, lb_loss_weight=1.0):
        super().__init__()
        self.edge_generator = SparseSoftHyperedgeGeneration(
            node_dim=embed_dim,
            num_dyn_hyperedges=num_dyn_hyperedges,
            k=top_k,
            num_fixed_hyperedges=num_fixed_hyperedges,
            num_heads=num_heads,
            dropout=dropout
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.lb_loss_weight = lb_loss_weight

    def forward(self, X):
        A = self.edge_generator(X) 
        lb_loss = 0
        if self.training:
            lb_loss = self.edge_generator.load_balance_loss() * self.lb_loss_weight

        He = torch.bmm(A.transpose(1, 2), X)  
        He = self.edge_proj(He)
        X_new = torch.bmm(A, He) 
        X_new = self.node_proj(X_new)
        return X_new + X, lb_loss
        

class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )
    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class ViTWithSoftHGNN_SeS(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, dim_head, mlp_dim, 
                 pool='cls', channels=3, dropout=0., emb_dropout=0.,
                 num_dyn_hyperedges=32, top_k=16, num_fixed_hyperedges=16, num_edge_head=8, lb_loss_weight=0.01
                 ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.soft_hgnn = SoftHGNN_SeS(
            embed_dim=dim,
            num_dyn_hyperedges=num_dyn_hyperedges,
            top_k=top_k,
            num_fixed_hyperedges=num_fixed_hyperedges,
            num_heads=num_edge_head,
            dropout=dropout,
            lb_loss_weight= lb_loss_weight
        )
        
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x, loss_lb  = self.soft_hgnn(x)
        x = x.mean(dim=1) if self.pool=='mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x), loss_lb



