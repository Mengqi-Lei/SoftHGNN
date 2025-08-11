import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseSoftHyperedgeGeneration(nn.Module):
    def __init__(self, node_dim, num_dyn_hyperedges, k, num_fixed_hyperedges=0,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_dyn_hyperedges = num_dyn_hyperedges  
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

        A = torch.softmax(logits, dim=1) 
        A = A * mask 

        self.last_mask = mask.detach()
        self.last_A = A.detach()
        return A

    def load_balance_loss(self):
        B = self.last_mask.shape[0]
        dynamic_mask = self.last_mask[:, :, self.num_fixed_hyperedges:]  
        global_activation = dynamic_mask.squeeze(1).mean(dim=0) 
        target = self.k / self.num_dyn_hyperedges
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
