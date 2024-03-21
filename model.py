import torch
import numpy as np
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn

import data

'''
'''
class CountPrediction(nn.Module):
    def __init__(self, cell_size, gene_size):
        super(CountPrediction, self).__init__()

        self.k = nn.Parameter(torch.rand(gene_size))
        self.d = nn.Parameter(torch.rand(gene_size))

        # Switching time and the u/s count at that time
        self.t0_g3 = nn.Parameter(torch.rand(gene_size))
        self.u0_g3 = nn.Parameter(torch.rand(gene_size))
        self.s0_g3 = nn.Parameter(torch.rand(gene_size))
        
        self.t = nn.Parameter(torch.rand(cell_size, gene_size))

    def S_trans(self, t):
        return 1 / (1 + torch.exp(-self.k * (t - self.t0_g3 - self.d)))
    
    def predict_u(self, alpha, beta, S):
        tilde_u = ((alpha / beta) * (1 - torch.exp(-beta * self.t)) * (1 - S) +
                   (alpha / beta) * S +
                   (self.u0_g3 * torch.exp(-beta * (self.t - self.t0_g3)) - (alpha / beta)) * S)
        
        return tilde_u
    
    def predict_s(self, alpha, beta, gamma, S):
        tilde_s = (((alpha / gamma) * (1 - torch.exp(-gamma * self.t)) +
                    (alpha / (gamma - beta)) * (torch.exp(-gamma * self.t) - torch.exp(-beta * self.t))) * (1 - S) +
                   (alpha / gamma) * S +
                   (beta * self.u0_g3 / (gamma - beta) * (torch.exp(-gamma * (self.t - self.t0_g3)) - torch.exp(-beta * (self.t - self.t0_g3)))) * S)

        return tilde_s

    def predict(self, out, t):
        # [num_cells, num_genes, 3] 
        alpha = out[:, :, 0] 
        gamma = out[:, :, 1]  
        beta = out[:, :, 2]   

        # Compute S_trans for each cell
        S = self.S_trans(self.t)

        # Compute tilde_u and tilde_s for each cell
        tilde_u = CountPrediction.predict_u(alpha, beta, S)
        tilde_s = CountPrediction.predict_s(alpha, beta, gamma, S)

        return tilde_u, tilde_s

    def forward(self, out, t):
        return self.predict(out)
    
'''
GAT Implementation
One GAT model for each gene, they are totally uncorrelated

Graph vertices: cells
Graph edges: similarity (xy distance, gene expression distance) between cells
    --option: kNN
    --option: ?
Input (vertices attributes): expression, u, s data for each cell
Output: alpha, beta, gamma for each cell
    --question: do we need t?
    --question: if t should be the same across genes, how to realize this restriction?
'''
class GAT(nn.Module):    
    def __init__(self, n_cell, n_gene, heads):
        # 3 out channels, alpha beta gamma (t?)
        super(GAT, self).__init__()
        self.n_cell, self.n_gene = n_cell, n_gene

        self.in_channel_size = 2 # or 3?
        self.mid_channel_size = 3
        # could test using more middle layer / different sizes?

        self.out_channel_size = 3
        self.heads = heads # number of attention heads
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.n_gene):
            self.convs.append(GATConv(self.in_channel_size, self.mid_channel_size, heads=heads, concat=True, dropout=0.6))
            self.bns.append(nn.BatchNorm1d(self.heads * self.mid_channel_size))

        # why use the same final layer?
        self.final_conv = GATConv(self.heads * self.mid_channel_size, self.out_channel_size, heads=1, concat=False, dropout=0.6)
        self.cp = CountPrediction(self.n_cell, self.n_gene)

    def predict(self, gene_list):
        outs = []
        # for each gene
        for i, gene_data in enumerate(gene_list):
            x, edge_index, edge_weight = gene_data.x, gene_data.edge_index, gene_data.edge_attr
            x = F.elu(self.convs[i](x, edge_index, edge_weight=edge_weight))
            x = self.bns[i](x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.final_conv(x, edge_index, edge_weight=edge_weight)
            outs.append(x)

        # Combine outputs for all genes, resulting in a tensor of shape [num_cells, num_genes, 3]
        out = torch.stack(outs, dim=1)
        return out
    
    def forward(self, gene_list):
        # alpha beta gamma's
        out = self.predict(gene_list)
        tilde_u, tilde_s = self.cp(out)
        return tilde_u, tilde_s

# Question - intuitive meaning of this loss?
# see VeloVi implementation at https://github.com/YosefLab/velovi/blob/main/velovi/_module.py#L488
def switch_loss(u_switch, s_switch, top_u, top_s):
    switch_loss = torch.sum((u_switch-top_u)**2 + (s_switch-top_s)**2)
    return switch_loss

def count_loss(pred_u, pred_s, u, s):
    count_loss = torch.mean((u-pred_u)**2 + (s-pred_s)**2)
    return count_loss

