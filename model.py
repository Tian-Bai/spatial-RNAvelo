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
    
class GAT(nn.Module):
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

        #TODO add equation predicting u/s
        return F.log_softmax(out, dim=-1)
    
        #TODO find a way to estimate t*
        #TODO : gene wise graph, forward returns equation is calculated in overleaf, get gene wise time switching time, loss will be spatially checked in neighborhood