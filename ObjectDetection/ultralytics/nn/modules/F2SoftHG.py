import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .conv import Conv
from .block import C3k, Bottleneck  

class SoftHyperedgeGeneration(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        
        self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)   
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape

        avg_context = X.mean(dim=1)           
        max_context, _ = X.max(dim=1)           
        context_cat = torch.cat([avg_context, max_context], dim=-1) 
        
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D) 
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets               
        
        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling  
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)  
        logits = self.dropout(logits) 
        return F.softmax(logits, dim=1)

class SoftHGNN(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.edge_generator = SoftHyperedgeGeneration(embed_dim, num_hyperedges, num_heads, dropout)

        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X) 

        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)

        X_new = torch.bmm(A, He) 
        X_new = self.node_proj(X_new)
        return X_new + X

class SoftHGBlock(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.softhgnn = SoftHGNN(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) 
        tokens = self.softhgnn(tokens)  
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out 
        
class FusionModule(nn.Module):
    def __init__(self, C, adjust_channels):
        super().__init__()
        
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if adjust_channels:
            self.conv_out = Conv(4 * C, C, 1)
        else:
            self.conv_out = Conv(3 * C, C, 1)
        
    def forward(self, x):
        x0_ds = self.downsample(x[0])
        x2_up = self.upsample(x[2])
        x_cat = torch.cat([x0_ds, x[1], x2_up], dim=1)
        out = self.conv_out(x_cat)
        return out

class F2SoftHG(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, shortcut=False, g=1, e=0.5, adjust_channels=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        self.fuse = FusionModule(c1, adjust_channels)
        self.softhgbranch1 = SoftHGBlock(embed_dim=self.c, 
                                   num_hyperedges=8, 
                                   num_heads=8,
                                   dropout=0.1)
        self.softhgbranch2 = SoftHGBlock(embed_dim=self.c, 
                                   num_hyperedges=8, 
                                   num_heads=8,
                                   dropout=0.1)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        softhg_out1 = self.softhgbranch1(y[1])
        softhg_out2 = self.softhgbranch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = softhg_out1
        y.append(softhg_out2)
        return self.cv2(torch.cat(y, 1))

class ShapeAlignConv(nn.Module):
    def __init__(self, in_channels, adjust_channels=True):
        super().__init__()
        self.adjust_channels = adjust_channels
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if adjust_channels:
            self.conv = Conv(in_channels, in_channels * 2, 1)
    
    def forward(self, x):
        x = self.downsample(x)
        if self.adjust_channels:
            x = self.conv(x)
        return x

class MergeConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Conv(in_channels * 2, in_channels, 1)
    def forward(self, x):
        x_cat = torch.cat(x, dim=1)
        return self.conv(x_cat)