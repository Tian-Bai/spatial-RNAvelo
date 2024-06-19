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
scv.pp.filter_and_normalize(st, min_shared_counts=20, n_top_genes=100)
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
We might consider using simpler techniques (e.g. PCA) when there are too few data to train
'''
# convert u, s to torch.tensor
u = torch.tensor(u)
s = torch.tensor(s)

RETRAIN_VAE = False

LATENT_DIM = 32
N, d = z.shape

# training the VAE for latent representation learning
# for model selection, see wandb. Could use 128-dim latent space, but the performance boost is not very significant
vae = model.VAE(in_channels=d, latent_dim=LATENT_DIM, hidden_dims=[64, 32], beta=3).double()

LOG_VAE = True and RETRAIN_VAE and LOG
VALIDATE_VAE = False

# it is found that this vae could generalize to unseen data. For the saved model, we need to train use all available samples.

xy_train, xy_valid, s_train, s_valid, u_train, u_valid = train_test_split(xy, s, u, train_size=0.8)
spatial_df_train = util.SpatialDataset(xy_train, s_train, u_train)

spatial_loader_train = DataLoader(spatial_df_train, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)

if RETRAIN_VAE:
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)

    if VALIDATE_VAE:
        xy_train, xy_valid, s_train, s_valid, u_train, u_valid = train_test_split(xy, s, u, train_size=0.8)
        spatial_df_train = util.SpatialDataset(xy_train, s_train, u_train)
        spatial_loader_train = DataLoader(spatial_df_train, batch_size=32, shuffle=True)

        for epoch in range(500):
            total_train_loss, total_recon_loss, total_kl_loss = 0, 0, 0

            for batch_idx, (xy_mini, s_mini, u_mini) in enumerate(spatial_loader_train):
                s_mini = s_mini.double()
                u_mini = u_mini.double()

                recon, input, mu, log_var = vae(torch.cat((s_mini, u_mini), dim=1))

                # mn_scale: dimension of latent layer / number of samples
                # as in beta-vae paper (2017), model with higher latent dimension need more constraint pressure (beta) 
                loss, recon_loss, kl_loss = vae.loss(recon, input, mu, log_var, mn_scale=16/len(spatial_df_train))
                total_train_loss += loss
                total_recon_loss += recon_loss
                total_kl_loss += kl_loss

                loss.backward()
                # to avoid gradient exploding
                torch.nn.utils.clip_grad_norm(vae.parameters(), 0.1)

                optimizer.step()

            recon, input, mu, log_var = vae(torch.cat((s_valid, u_valid), dim=1))
            valid_loss, valid_recon_loss, valid_kl_loss = vae.loss(recon, input, mu, log_var, mn_scale=1)

            print(f'epoch {epoch}, Train loss: {total_train_loss}')
            print(f'epoch {epoch}, Test loss: {valid_loss.item()}')

            if LOG_VAE:
                wandb.log({'VAE training loss': total_train_loss.item(),
                        'VAE training recon loss': total_recon_loss.item(),
                        'VAE training KL loss': total_kl_loss.item(),
                        'VAE validation loss': valid_loss.item(),
                        'VAE validation recon loss': valid_recon_loss.item(),
                        'VAE validation KL loss': valid_kl_loss.item()})
            # time.sleep(0.2)
    else:
        spatial_df = util.SpatialDataset(xy, s, u)
        spatial_loader = DataLoader(spatial_df, batch_size=32, shuffle=True)

        for epoch in range(500):
            total_train_loss, total_recon_loss, total_kl_loss = 0, 0, 0

            for batch_idx, (xy_mini, s_mini, u_mini) in enumerate(spatial_loader):
                s_mini = s_mini.double()
                u_mini = u_mini.double()

                recon, input, mu, log_var = vae(torch.cat((s_mini, u_mini), dim=1))

                # mn_scale: dimension of latent layer / number of samples
                # as in beta-vae paper (2017), model with higher latent dimension need more constraint pressure (beta) 
                loss, recon_loss, kl_loss = vae.loss(recon, input, mu, log_var, mn_scale=16/len(spatial_df))
                total_train_loss += loss
                total_recon_loss += recon_loss
                total_kl_loss += kl_loss

                loss.backward()
                # to avoid gradient exploding
                torch.nn.utils.clip_grad_norm(vae.parameters(), 0.1)

                optimizer.step()

            print(f'epoch {epoch}, Train loss: {total_train_loss}')

            if LOG_VAE:
                wandb.log({'VAE training loss': total_train_loss.item(),
                        'VAE training recon loss': total_recon_loss.item(),
                        'VAE training KL loss': total_kl_loss.item()})
    torch.save(vae.state_dict(), 'vae.pth')
else:
    vae.load_state_dict(torch.load('vae.pth'))

# after training, get the corresponding latent representations
vae.eval()
with torch.no_grad():
    latent, _ = vae.encode(torch.cat((s, u), dim=1))

# for a further visualization
VAE_VIS = False

if VAE_VIS:
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
The current model might still contain too many parameters - we have only ~1k training samples. Consider training separate models, each for one gene?
Or - doing another dimension reduction for the output (gradient)?

Currently it seems the model is not learning anything meaningful.
1. A MLP with hidden layer [16, 8] does not perform better than a MLP with hidden layer [2]
2. (?) Most predicted value are zero (but so are the original data)

GAT learning outcome is nearly random (varies a lot). Even for a low-dimensional task (20), the model is still not learning anything, 
suggesting that this might not be a dimensionality problem (at least for now).
MLP is also random.
'''
# focusing on x direction first
RETRAIN_MODEL = True
LOG_MODEL = True and LOG and RETRAIN_MODEL
VALIDATE_MODEL = False

MODEL = 'MLP'

# indexes of xgrad where the element is not NaN
xgrad_notNaN = np.where(~np.isnan(xgrad[:, 0]))[0]
ygrad_notNaN = np.where(~np.isnan(ygrad[:, 0]))[0]

# edges
edge_index = gnn.knn_graph(torch.tensor(xy), k=5, flow='source_to_target')
edge_index = torch.cat([edge_index, edge_index.flip(dims=[0])], dim=1).unique(dim=1)

if MODEL == 'MLP':
    x_model = model.MLP_regression(2 + LATENT_DIM, hidden_dims=[16, 8], out_channels=len(xgrad[0])).double()
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-4)

    if RETRAIN_MODEL:
        x_model.train()
        if VALIDATE_MODEL:
            xy_train, xy_valid, latent_train, latent_valid, x_spatial_grad_train, x_spatial_grad_valid = train_test_split(xy[xgrad_notNaN], latent[xgrad_notNaN], xgrad[xgrad_notNaN], train_size=0.8)
            x_spatial_grad_df_train = util.SpatialGradientDataset(xy_train, latent_train, x_spatial_grad_train)
            x_spatial_grad_loader_train = DataLoader(x_spatial_grad_df_train, batch_size=32, shuffle=True)
            
            for epoch in range(1000):
                x_loss = 0

                for batch_idx, (xy_train, latent_train, xgrad_train) in enumerate(x_spatial_grad_loader_train):
                    latent_train = latent_train.double()
                    xgrad_train = xgrad_train.double()

                    x_predgrad = x_model(torch.cat((xy_train, latent_train), dim=1))
                    loss = F.mse_loss(xgrad_train, x_predgrad)
                    x_loss += loss

                    loss.backward()
                    x_optimizer.step()
                valid_x_predgrad = x_model(torch.cat((torch.tensor(xy_valid), latent_valid), dim=1))
                valid_x_loss = F.mse_loss(torch.tensor(x_spatial_grad_valid), valid_x_predgrad)

                print(f'epoch {epoch}, Train loss (x): {x_loss}')
                print(f'epoch {epoch}, Validation loss (x): {valid_x_loss}')
                
                if LOG_MODEL:
                    wandb.log({'xModel training loss': x_loss.item(),
                               'xModel validation loss': valid_x_loss.item()})
        else:
            x_spatial_grad_df = util.SpatialGradientDataset(xy[xgrad_notNaN], latent[xgrad_notNaN], xgrad[xgrad_notNaN])
            x_spatial_grad_loader = DataLoader(x_spatial_grad_df, batch_size=32, shuffle=True)

            for epoch in range(1000):
                x_loss = 0

                for batch_idx, (xy_train, latent_train, xgrad_train) in enumerate(x_spatial_grad_loader):
                    latent_train = latent_train.double()
                    xgrad_train = xgrad_train.double()

                    x_predgrad = x_model(torch.cat((xy_train, latent_train), dim=1))
                    loss = F.mse_loss(xgrad_train, x_predgrad)
                    x_loss += loss

                    loss.backward()
                    x_optimizer.step()

                print(f'epoch {epoch}, Total loss (x): {x_loss}')

                if LOG_MODEL:
                    wandb.log({'xModel training loss': x_loss.item()})
        torch.save(x_model.state_dict(), 'xmlp.pth')
    else:
        x_model.load_state_dict(torch.load('xmlp.pth'))

    x_model.eval()
    with torch.no_grad():
        x_predgrad = x_model(torch.cat((torch.tensor(xy), latent), dim=1))

elif MODEL == 'GAT':
    x_model = model.GAT_interpolation(2 + LATENT_DIM, hidden_dims=[32, 16], out_channels=len(xgrad[0])).double()
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-3)

    if RETRAIN_MODEL:
        x_model.train()
        if VALIDATE_MODEL:
            pass # to be implemented. Use another mask for validation
        else:
            x_spatial_grad_df = util.SpatialGradientDataset(xy, latent, xgrad)
            x_spatial_grad_loader = DataLoader(x_spatial_grad_df, batch_size=32, shuffle=True)

            for epoch in range(300): # here we use the whole batch
                x_loss = 0

                x_predgrad = x_model(torch.cat((torch.tensor(xy), latent), dim=1), edge_index)
                loss = F.mse_loss(torch.tensor(xgrad)[xgrad_notNaN], x_predgrad[xgrad_notNaN])
                x_loss += loss

                loss.backward()
                x_optimizer.step()

                print(f'epoch {epoch}, Total loss (x): {x_loss}')

                if LOG_MODEL:
                    wandb.log({'xModel training loss': x_loss.item()})
        torch.save(x_model.state_dict(), 'xmlp.pth')
    else:
        x_model.load_state_dict(torch.load('xmlp.pth'))

    x_model.eval()
    with torch.no_grad():
        x_predgrad = x_model(torch.cat((torch.tensor(xy), latent), dim=1), edge_index)

plt.figure()
plt.quiver(xy[:, 0], xy[:, 1], x_predgrad[:, 0], x_predgrad[:, 1])
plt.show()

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