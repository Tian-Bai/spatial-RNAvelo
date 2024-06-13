import torch
import numpy as np
from torch_geometric import nn as gnn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
from torch.utils.data import DataLoader
import math
from torch.utils.data import Dataset
import scanpy as sc
import scvelo as scv
import pandas as pd
from matplotlib import pyplot as plt
import model, util
import time
from sklearn.model_selection import train_test_split
import wandb
from sklearn.manifold import TSNE

LOG = False

if LOG:
    wandb.init(project='GAT-rna-velo', config={})

''' 
Part 1: load, preprocessing, discretization and numerical estimation of gradient
'''

scrna_path = "chicken_heart\\RNA_D14_adata.h5ad"
st_path = "chicken_heart\\Visium_D14_adata.h5ad"

st = scv.read(st_path)
scv.pp.filter_and_normalize(st, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(st, n_pcs=30, n_neighbors=30)

u = st.to_df('unspliced').to_numpy()
s = st.to_df('spliced').to_numpy()

xy = st.obsm['X_xy_loc']
z = np.column_stack((s, u))
A, xy_new, _, _ = util.discretize(xy, z, 180)

xgrad, ygrad, Dx, Dy = util.numeric_gradient(xy_new, A) 

# plot A, Dx and Dy
# fig, axs = plt.subplots(figsize=(20, 13), nrows = 2, ncols = 2)

# xx, yy = np.where(~np.isnan(A[:, :, 0]))
# axs[0, 0].scatter(xx, yy, s=3)

# xx, yy = np.where(~np.isnan(Dx[:, :, 0]))
# axs[0, 1].scatter(xx, yy, s=3)

# xx, yy = np.where(~np.isnan(Dy[:, :, 0]))
# axs[1, 0].scatter(xx, yy, s=3)
# plt.show()

''' 
Part 2: train the VAE
'''
# convert u, s to torch.tensor
u = torch.tensor(u)
s = torch.tensor(s)

RETRAIN = False

LATENT_DIM = 16
N, d = z.shape

# training the VAE for latent representation learning
# for model selection, see wandb. Could use 128-dim latent space, but the performance boost is not very significant
vae = model.VAE(in_channels=d, latent_dim=LATENT_DIM, hidden_dims=[64, 32], beta=3).double()

LOG_VAE = True and LOG

spatial_df = util.SpatialDataset(xy, s, u)
spatial_loader = DataLoader(spatial_df, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)

if RETRAIN:
    vae.train()
    for epoch in range(300):
        total_train_loss, total_recon_loss, total_kl_loss = 0, 0, 0

        for batch_idx, (xy, s, u) in enumerate(spatial_loader):
            s = s.double()
            u = u.double()

            recon, input, mu, log_var = vae(torch.cat((s, u), dim=1))
            loss, recon_loss, kl_loss = vae.loss(recon, input, mu, log_var, mn_scale=32/N)
            total_train_loss += loss
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss

            loss.backward()
            # to avoid gradient exploding
            torch.nn.utils.clip_grad_norm(vae.parameters(), 0.1)

            optimizer.step()

        print(f'epoch {epoch}, Total loss: {total_train_loss}')

        if LOG_VAE:
            wandb.log({'VAE training loss': total_train_loss.item(),
                    'VAE recon loss': total_recon_loss.item(),
                    'VAE KL loss': total_kl_loss.item()})
        # time.sleep(0.2)
    torch.save(vae.state_dict(), 'vae.pth')
else:
    vae.load_state_dict(torch.load('vae.pth'))

    # after training, get the corresponding latent representations
vae.eval()
with torch.no_grad():
    latent, _, _, _ = vae(torch.cat((s, u), dim=1))

# for a further visualization
tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(latent.numpy())

tsne = TSNE(n_components=2, random_state=42)
original_tsne = tsne.fit_transform(torch.cat((s, u), dim=1).numpy())

fig, axs = plt.subplots(figsize=(20, 10), ncols=2)

axs[0].scatter(latent_tsne[:, 0], latent_tsne[:, 1], s=3)
axs[1].scatter(original_tsne[:, 0], original_tsne[:, 1], s=3)
plt.show()

''' 
Part 3: train the gradient prediction model
'''
LOG_MLP = True and LOG

# indexes of xgrad where the element is not NaN
xgrad_notNaN = np.where(~np.isnan(xgrad[:, 0]))[0]
ygrad_notNaN = np.where(~np.isnan(ygrad[:, 0]))[0]

x_spatial_grad_df = util.SpatialGradientDataset(xy[xgrad_notNaN], latent[xgrad_notNaN], xgrad[xgrad_notNaN])
y_spatial_grad_df = util.SpatialGradientDataset(xy[ygrad_notNaN], latent[xgrad_notNaN], ygrad[ygrad_notNaN])

x_spatial_grad_loader = DataLoader(x_spatial_grad_df, batch_size=32, shuffle=True)
y_spatial_grad_loader = DataLoader(y_spatial_grad_df, batch_size=32, shuffle=True)

# an example of using only MLP
x_mlp = model.MLP_regression(2 + LATENT_DIM, hidden_dims=[64, 64], out_channels=len(xgrad[0])).double()
y_mlp = model.MLP_regression(2 + LATENT_DIM, hidden_dims=[64, 64], out_channels=len(ygrad[0])).double()

mse_loss = nn.MSELoss()

x_optimizer = torch.optim.Adam(x_mlp.parameters(), lr=5e-4)
y_optimizer = torch.optim.Adam(y_mlp.parameters(), lr=5e-4)

x_mlp.train()
for epoch in range(300):
    x_loss = 0

    for batch_idx, (xy, latent, xgrad) in enumerate(x_spatial_grad_loader):
        latent = latent.double()
        xgrad = xgrad.double()

        x_predgrad = x_mlp(torch.cat((xy, latent), dim=1))
        loss = mse_loss(xgrad, x_predgrad)
        x_loss += loss

        loss.backward()
        x_optimizer.step()

    print(f'epoch {epoch}, Total loss (x): {x_loss}')

    if LOG_MLP:
        wandb.log({'MLP for xgrad loss': x_loss.item()})

# y_mlp.train()
# for epoch in range(300):
#     y_loss = 0

#     for batch_idx, (xy, latent, ygrad) in enumerate(y_spatial_grad_loader):
#         latent = latent.double()
#         ygrad = ygrad.double()

#         y_predgrad = y_mlp(torch.cat((xy, latent), dim=1))
#         loss = mse_loss(ygrad, y_predgrad)
#         y_loss += loss

#         loss.backward()
#         y_optimizer.step()

#     print(f'epoch {epoch}, Total loss (x): {y_loss}')

#     if LOG_MLP:
#         wandb.log({'MLP for ygrad loss': y_loss.item()})