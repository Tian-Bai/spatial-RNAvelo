import torch
from torch.utils.data import Dataset
import scanpy as sc
import scvelo as scv
import pandas as pd
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch import nn

class SingleCellDataset(Dataset):
    def __init__(self, adata_path, filter_normalize = True):
        self.adata = sc.read(adata_path)



        scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(self.adata, n_neighbors=30, n_pcs=30)

        self.unspliced = pd.DataFrame(self.adata.layers['unspliced'],columns=self.adata.to_df().columns)
        self.spliced = pd.DataFrame(self.adata.layers['spliced'],columns=self.adata.to_df().columns)

        #Add similarity matrix


    def __len__(self):
        return self.unspliced.shape[0]
    
    def __getitem__(self, index):
        cell_data = torch.stack([self.unspliced[index, :], self.spliced[index, :]], dim=1)
        return cell_data
    
class CountPrediction(nn.Module):
    def __init__(self, cell_size, gene_size):
        super(CountPrediction, self).__init__()

        #TODO: Change size to number of genes
        self.k = nn.Parameter(torch.randn(gene_size))
        self.d = nn.Parameter(torch.randn(gene_size))
        self.t0_g3 = nn.Parameter(torch.randn(gene_size))
        
        #remove and change to function call
        self.u0_g3 = nn.Parameter(torch.randn(gene_size))
        self.t = nn.Parameter(torch.randn(cell_size, gene_size))

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
        S = self.S_trans(t)

        # Compute tilde_u and tilde_s for each cell
        tilde_u = CountPrediction.predict_u(alpha, beta, S)
        tilde_s = CountPrediction.predict_s(alpha, beta, gamma, S)

        return tilde_u, tilde_s

    def forward(self, out, t):
        return self.predict(out)
    

class GAT(nn.Module):
    def softmax_time():
        return
    def __init__(self, num_genes, out_channels=4):
        #4 out channels, alpha beta gamma and time t
        super(GAT, self).__init__()
        self.num_genes = num_genes
        self.in_channels = 2
        self.out_channels = out_channels
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_genes):
            self.convs.append(GATConv(self.in_channels, 8, heads=8, dropout=0.6, concat=True))
            self.bns.append(nn.BatchNorm1d(8 * 8))

        self.final_conv = GATConv(8 * 8, self.out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data_list):
        outs = []
        for i, data in enumerate(data_list):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.elu(self.convs[i](x, edge_index, edge_weight=edge_weight))
            x = self.bns[i](x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.final_conv(x, edge_index, edge_weight=edge_weight)
            outs.append(x)

        # Combine outputs for all genes, resulting in a tensor of shape [num_cells, num_genes, 3]
        out = torch.stack(outs, dim=1)

        return 