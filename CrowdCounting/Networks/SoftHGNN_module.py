# ---------------------------
# SoftHGNN 
# ---------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gate = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, X):
        A = self.edge_generator(X) 
        He = torch.bmm(A.transpose(1, 2), X)  
        He = self.edge_proj(He)
        X_new = torch.bmm(A, He) 
        X_new = self.node_proj(X_new)
        
        return X + self.gate * X_new
