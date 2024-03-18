import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import scvelo as scv
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.metrics import pairwise_distances
import hnswlib
import dgl

class SingleCellDataset(Dataset):
    def __init__(self, adata_path, filter_normalize = True):
        # assuming adata has scrna and st data integrated together
        self.adata = sc.read(adata_path)

        if filter_normalize:
            scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(self.adata, n_neighbors=30, n_pcs=30)

        # seems cannot directly do 'pd.dataframe' on sparse matrix (?); will give a shape mismatch error

        self.unspliced = self.adata.to_df(layer='unspliced')
        self.spliced = self.adata.to_df(layer='spliced')
        self.xy = self.adata.obs[['array_row', 'array_col']] 
        self.n_cell, self.n_gene = self.spliced.shape
        # the name could depend on actual dataset

        # self.unspliced = pd.DataFrame(self.adata.layers['unspliced'], columns=self.adata.to_df().columns)
        # self.spliced = pd.DataFrame(self.adata.layers['spliced'], columns=self.adata.to_df().columns)

        self.get_graph()

    def get_graph(self, k=50, c_expr=0.2, c_xy=0.8):
        # using mixed distance of expression and xy, and then do kNN graph
        # NOTE: what could be a good choice of c?

        self.g = dgl.graph()
        self.g.add_nodes(self.n_cell)
        edge_list = []

        # if cell number is small, could brute force
        if self.n_cell < 3000:
            expr_dist = pairwise_distances(self.spliced)
            xy_dist = pairwise_distances(self.xy)
            dist = c_expr * expr_dist + c_xy * xy_dist
            close_ind = np.argsort(dist, axis=1)[:, :k] 
        else: # could use hnswlib
            # NOTE: should we use pca first? n_gene could be huge
            # NOTE: this does not support mixed distance
            knn = hnswlib.Index(space="l2", dim=self.n_gene)
            knn.init_index(max_elements=self.n_cell, ef_construction=200, M=30) # the later 2 are coef for runtime/accuracy optims
            knn.add_items(self.xy)
            knn.set_ef(k)
            close_ind = knn.knn_query(self.xy, k=k)[0].astype(int)

        for i in range(close_ind.shape[0]):
            for j in close_ind[i]:
                edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        self.g.add_edges(src, dst)
        self.g.add_edges(dst, src)

    def create_torch_geometric_data(self, edges, gene_index):
        # Convert edges to a PyTorch tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        unspliced_gene = self.unspliced[:, gene_index] 
        spliced_gene = self.spliced[:, gene_index]

        node_features = torch.stack((unspliced_gene, spliced_gene), dim=1) 

        data = Data(x=node_features, edge_index=edge_index)

        return data

    
    def __len__(self):
        return self.unspliced.shape[0]
    
    def __getitem__(self, index):
        cell_data = torch.stack([self.unspliced[index, :], self.spliced[index, :]], dim=1)
        return cell_data
    
class SingleCellLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        super().__init__(dataset, batch_size, shuffle)

    # TODO: train/validation sampler
    def sampler():
        pass