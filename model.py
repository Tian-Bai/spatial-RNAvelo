import torch
import numpy as np
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
import math
from torch.utils.data import Dataset
import scanpy as sc
import scvelo as scv
import pandas as pd
from matplotlib import pyplot as plt

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim=32, hidden_dims=[64, 32], beta=4):
        super(nn.Module, self).__init__()
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
        for i in range(len(all_dim), 0, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels=all_dim[i], out_features=all_dim[i-1]),
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
    
    def loss(self, recon, input, mu, log_var):
        recon_loss = F.mse_loss(recon, input)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss
    
class MLP_regression(nn.Module):
    pass

class GAT_interpolation(nn.Module):
    pass