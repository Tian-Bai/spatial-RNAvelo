import data
import torch.optim as optim
import torch
import model 

num_epochs = 10

def train(dataset):
    dataloader = data.SingleCellLoader(dataset, batch_size=1, shuffle=False)
    model_gat = model.GAT(n_cell=dataset.n_cell, n_gene=len(dataset), heads=8)
    optimizer = optim.Adam(model_gat.parameters(), lr=0.001)


    for epoch in range(num_epochs):
        for gene_index in range(len(dataset)):
            #for data_cell in dataloader: 

            optimizer.zero_grad()
            gene_graph = dataset.create_torch_geometric_data(gene_index)
            u = dataset.get_cells_by_gene(gene_index)[:, 0]
            s = dataset.get_cells_by_gene(gene_index)[:, 1]
            tilde_u, tilde_s = model_gat(gene_graph, gene_index)

            count_loss_value = model.count_loss(tilde_u, tilde_s, u, s)

            threshold_u = torch.quantile(u.flatten(), 0.95)
            threshold_s = torch.quantile(s.flatten(), 0.95)

            top_u = torch.mean(u[u > threshold_u])
            top_s = torch.mean(s[s > threshold_s])

            switch_loss_value = model.switch_loss(model_gat.cp.u0_g3, model_gat.cp.s0_g3, top_u, top_s)
            total_loss = count_loss_value + switch_loss_value
            print(total_loss)
            total_loss.backward()
            optimizer.step()

if __name__ == "__main__":
    scrna_path = "..\\SIRV Datasets\\Chicken_heart\\RNA_D14_adata.h5ad"
    st_path = "C:\\Users\\yasmi\\Desktop\\research\\SIRV Datasets\\Chicken_heart\\Visium_D14_adata.h5ad"

    dataset = data.SingleCellDataset(st_path)
    dataset.process_graph()
    train(dataset)
