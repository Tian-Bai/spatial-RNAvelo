import torch
import numpy as np
from torch_geometric import nn as gnn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
import math
from torch.utils.data import Dataset
import scanpy as sc
import scvelo as scv
import pandas as pd
from matplotlib import pyplot as plt

''' 
in-channels: number of genes * 2 = dimension of spliced/unspliced conbined
latent_dim: dimension of latent representation
need to train VAE first
'''
class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim=32, hidden_dims=[64, 32], beta=4):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta

        all_dim = [in_channels] + hidden_dims + [latent_dim]
        
        # encoder
        # in_channel is used as a temporary variable here
        modules = []
        for i in range(len(all_dim) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=all_dim[i], out_features=all_dim[i+1]), 
                    nn.LeakyReLU()
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.f_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.f_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder
        modules = []
        for i in range(len(all_dim) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=all_dim[i], out_features=all_dim[i-1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        res = self.encoder(input)
        mu = self.f_mu(res)
        log_var = self.f_log_var(res)
        return mu, log_var
    
    def decode(self, z):
        res = self.decoder(z)
        return res
    
    def reparam(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparam(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    def loss(self, recon, input, mu, log_var, mn_scale):
        recon_loss = F.mse_loss(recon, input)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recon_loss + self.beta * mn_scale * kl_loss
        return loss, recon_loss, kl_loss
    
''' 
in_channels: 2 + dimension of VAE low-dim representation
out_channels: dimension of gradient representation
'''
class MLP_regression(nn.Module):
    def __init__(self, in_channels, hidden_dims=[64, 64], out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        all_dim = [in_channels] + hidden_dims + [out_channels]
        modules = []
        for i in range(len(all_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=all_dim[i], out_features=all_dim[i+1]),
                    # batchnorm?
                    nn.BatchNorm1d(num_features=all_dim[i+1]),
                    nn.ReLU()
                )
            )
        self.mlp = nn.Sequential(*modules)

    def forward(self, input):
        return self.mlp(input)

'''
in_channels: dimension of VAE low-dim representation
graph structure: from x-y data
out_channels: dimension of gradient representation
by default, use 1-head attention.
'''
class GAT_interpolation(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 32], out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        all_dim = [in_channels] + hidden_dims 

        # GAT part
        modules = []
        for i in range(len(all_dim) - 1):
            modules.append((gnn.GATConv(all_dim[i], all_dim[i+1], heads=1, dropout=0.5), 'x, edge_index -> x'))
            modules.append(nn.ReLU())

        self.gat = gnn.Sequential('x, edge_index', modules)
        self.final_linear = nn.Linear(hidden_dims[-1], out_channels)

    def forward(self, x, edge_index):
        res = self.gat(x, edge_index)
        return self.final_linear(res)
    
''' 
in_channel: dimension of VAE low-dim representation
out_channel: number of genes * 4 (4 gradients for each gene)
latent_dim: dimension of gradient representation
'''
class GradientModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=32):
        super(nn.Module, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.regression = MLP_regression(self.in_channels, out_channels=latent_dim)
        self.interpolation = GAT_interpolation(self.in_channels, out_channels=latent_dim)

        # for averaging
        self.alphas = torch.full(latent_dim, fill_value=0.5)
        self.final_linear = nn.Sequential(
            nn.Linear(latent_dim, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        regre = self.regression(x)
        interp = self.interpolation(x, edge_index)
        out = regre * self.alphas + interp * (1 - self.alphas)
        return self.final_linear(out)