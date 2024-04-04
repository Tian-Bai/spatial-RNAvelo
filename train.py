import data
import torch.optim as optim
import torch
import model
import matplotlib.pyplot as plt
import wandb
import numpy as np

wb = True
num_epochs = 100000
lr = 1e-4

def train(dataset):
    dataloader = data.SingleCellLoader(dataset, batch_size=1, shuffle=False)
    model_gat = model.GAT(n_cell=dataset.n_cell, n_gene=len(dataset), layers=32, heads=2)
    optimizer = optim.Adam(model_gat.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        for gene_index in range(len(dataset)):
            # only test one gene:
            if gene_index != 0:
                break

            optimizer.zero_grad()
            gene_graph = dataset.create_torch_geometric_data(gene_index)
            u = dataset.get_cells_by_gene(gene_index)[:, 0]
            s = dataset.get_cells_by_gene(gene_index)[:, 1]
            tilde_u, tilde_s = model_gat(gene_graph, gene_index)
            print(f"shapes: {tilde_u.shape}, {u.shape}")

            count_loss_value = model.count_loss(tilde_u, tilde_s, u, s)
            if torch.isnan(count_loss_value):
                print("Nan detected.")
                # np.savetxt('predicted_u.txt', tilde_u.detach().numpy())
                # np.savetxt('predicted_s.txt', tilde_s.detach().numpy())
                # np.savetxt('u.txt', u.detach().numpy())
                # np.savetxt('s.txt', s.detach().numpy())
                exit(-1)

            threshold_u = torch.quantile(u.flatten(), 0.95)
            threshold_s = torch.quantile(s.flatten(), 0.95)

            top_u = torch.mean(u[u > threshold_u])
            top_s = torch.mean(s[s > threshold_s])

            switch_loss_value = model.switch_loss(model_gat.cp.u0_g3, model_gat.cp.s0_g3, top_u, top_s)
            # switch_loss_value = 0
            total_loss = count_loss_value + switch_loss_value

            if wb:
                wandb.log({"loss": total_loss})
            else:
                print(f"loss: {total_loss}")

            total_loss.backward()
            optimizer.step()

if __name__ == "__main__":
    scrna_path = "chicken_heart\\RNA_D14_adata.h5ad"
    st_path = "chicken_heart\\Visium_D14_adata.h5ad"

    if wb:
        wandb.init(project="GAT-rna-velo", config={
            "learning_rate": lr,
            "architecture": "CNN",
            "epochs": num_epochs
        })
    dataset = data.SingleCellDataset(st_path)

    dataset.process_graph()
    train(dataset)
    if wb:
        wandb.finish()
