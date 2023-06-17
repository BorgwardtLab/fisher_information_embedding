import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize" : (10, 6),
    "font.size": 22
})


def train_epoch(dset, model, optimizer):
    model.train()

    tic = timer()
    optimizer.zero_grad()
    output = model(dset)
    output = output[dset.train_mask]
    labels = dset.y[dset.train_mask]
    loss = F.cross_entropy(output, labels)
    loss.backward()
    optimizer.step()
    toc = timer()

    preds = output.data.argmax(dim=1)
    train_acc = torch.sum(preds == labels).item() / len(labels)

    print("Train loss: {:.4f} Acc: {:.4f} time: {:2f}s".format(
        loss.item(), train_acc, toc - tic))
    return loss.item()

@torch.no_grad()
def eval_epoch(dset, model, split='Val'):
    model.eval()

    mask = dset.val_mask if split == 'Val' else dset.test_mask

    tic = timer()
    output = model(dset)
    output = output[mask]
    labels = dset.y[mask]
    loss = F.cross_entropy(output, labels)
    toc = timer()

    preds = output.data.argmax(dim=1)
    val_acc = torch.sum(preds == labels).item() / len(labels)

    print("{} loss: {:.4f} Acc: {:.4f} time: {:2f}s".format(
          split, loss.item(), val_acc, toc - tic))
    return loss.item(), val_acc





class GAT(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, hidden_dim=8, heads=8, num_layers=2):
        super(GAT, self).__init__()
        self.hid = hidden_dim
        self.heads = heads
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.num_layers = num_layers

        layers = [GATConv(self.num_input_features, self.hid, heads=self.heads, dropout=0.6)]
        for i in range(num_layers-2):
            layers.append(
                GATConv(self.hid*self.heads, self.hid, heads=self.heads, dropout=0.6)
            )
        layers.append(GATConv(self.hid*self.heads, self.num_output_features, concat=False,
                           heads=1, dropout=0.6))
        self.layers = nn.ModuleList(layers)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        outputs = [x]
        for i in range(self.num_layers):
            x = F.dropout(outputs[-1], p=0.6, training=self.training)
            output = self.layers[i](x, edge_index)
            if i < self.num_layers -1:
                output = F.elu(output)
            outputs.append(output)

        return outputs[-2]
        #output = outputs[-1]
        #return F.log_softmax(output, dim=1)

    @torch.no_grad()
    def inference_ns(self, x_all, subgraph_loader):
        device = x_all.device
        out = [x_all.cpu()]
        for i, mod in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = mod((x, x_target), edge_index)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
            out.append(x_all.cpu())

        out = out[-1]
        return out


def load_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised Fisher information embedding for citation datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='name of dataset')

    # Model hyperparameters
    parser.add_argument('--heads', type=int, default=1,
                        help='heads')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='hidden size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')

    parser.add_argument('--outdir', type=str, default=None,
                        help='output directory')

    args = parser.parse_args()

    if args.outdir is not None:
        args.outdir = args.outdir + \
            f'/{args.dataset}' + \
            f'/{args.num_layers}_{args.hidden_size}_{args.heads}' + \
            f'/seed{args.seed}'
        os.makedirs(args.outdir, exist_ok=True)

    args.device = torch.device(torch.cuda.current_device()) \
        if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args



def main():
    global args
    args = load_args()
    print(args)

    name_data = args.dataset
    dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
    dataset.transform = T.NormalizeFeatures()


    model = GAT(dataset.num_node_features, dataset.num_classes,
                    hidden_dim=args.hidden_size, heads=args.heads, num_layers=args.num_layers
                    ).to(args.device)
    data = dataset[0].to(args.device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-4)

    best_val_acc = 0
    best_epoch = 0
    tic = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(
            epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_epoch(data, model, optimizer)
        val_loss, val_acc = eval_epoch(data, model, split='Val')
        lr_scheduler.step(val_acc)
        eval_epoch(data, model, split='Test')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    run_time = timer() - tic
    model.load_state_dict(best_weights)

    output = model(data)
    output = output.cpu().detach().numpy()
    print(output.shape)

    node_labels = data.y.cpu().detach().numpy()
    num_classes = len(set(node_labels))
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(output)


    fig = plt.figure()  # otherwise plots are really small in Jupyter Notebook
    for class_id in range(num_classes):
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20)

    #plt.legend()
    #plt.show()
    plt.title("GAT")
    plt.xticks([],[])
    plt.yticks([],[])
    fig.tight_layout()
    fig.savefig("tsne_gat.pdf")


if __name__ == '__main__':
    main()
