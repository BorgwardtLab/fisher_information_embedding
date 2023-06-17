# -*- coding: utf-8 -*-
import os
import copy
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric import datasets
from torch_geometric import utils
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import argparse
import pandas as pd
from timeit import default_timer as timer

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from fie.layers import FIELayer, FIEPooling, KernelLayer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize" : (10, 6),
    "font.size": 22
})





class SupFIENet(nn.Module):
    def __init__(self, num_class, input_size, num_layers=2,
                 hidden_size=64, num_mixtures=1, num_heads=1,
                 residue=True, out_proj='relu', out_proj_args=0.5, use_deg=False,
                 pooling='fie', concat=False):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.concat = concat

        # self.in_head = nn.Linear(input_size, hidden_size)
        self.in_head = KernelLayer(input_size, hidden_size, sigma=out_proj_args)

        layers = []

        for i in range(num_layers):
            layers.append(
                FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
                         residue, 'kernel', 'exp', out_proj_args, use_deg)
            )

        # layers.append(
        #     FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
        #              residue, 'kernel', 'exp', out_proj_args, use_deg)
        # )

        self.layers = nn.ModuleList(layers)

        if pooling == 'mean':
            self.pooling = gnn.global_mean_pool
        elif pooling == 'sum':
            self.pooling = gnn.global_add_pool
        elif pooling == 'fie':
            pool_size = hidden_size * num_layers + input_size if concat else hidden_size
            self.pooling = FIEPooling(
                pool_size, hidden_size, num_heads, residue, out_proj, out_proj_args
            )
        else:
            self.pooling = None

        output_size = hidden_size * (num_layers + 1) if concat else hidden_size
        self.classifier = nn.Linear(output_size, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        deg_sqrt = getattr(data, 'deg_sqrt', None)

        x = self.in_head(x)

        outputs = [x]
        output = x
        for i, mod in enumerate(self.layers):
            output = mod(output, edge_index, edge_attr, deg_sqrt=deg_sqrt)
            outputs.append(output)

        return outputs[-1]
        #if self.pooling is not None:
        #    output = self.pooling(output, data.batch)
        #return self.classifier(output)

    @torch.no_grad()
    def predict(self, data_loader, device='cpu'):
        self.eval()

        outputs = []
        for data in data_loader:
            data = data.to(device)
            outputs.append(self(data).cpu())

        outputs = torch.cat(outputs)
        return outputs

    def representation(self, n, x, edge_index, edge_attr=None, deg_sqrt=None,
                       before_out_proj=False):
        x = self.in_head(x)
        if n == -1:
            n = self.num_layers
        for i in range(n):
            x = self.layers[i](x, edge_index, edge_attr, deg_sqrt=deg_sqrt)
        if before_out_proj:
            x = self.layers[n](x, edge_index, edge_attr, deg_sqrt=deg_sqrt,
                               before_out_proj=before_out_proj)
        return x

    @torch.no_grad()
    def unsup_train(self, data_loader, n_samples=100000,
                    init=None,  device='cpu'):
        self.train(False)
        try:
            n_samples_per_batch = (n_samples + len(data_loader) - 1) // len(data_loader)
        except Exception:
            n_samples_per_batch = 1000
        n_sampled = 0
        samples = torch.Tensor(n_samples, self.input_size).to(device)

        for data in data_loader:
            data = data.to(device)
            x = data.x
            samples_batch = self.in_head.sample(x, n_samples_per_batch)
            size = min(samples_batch.shape[0], n_samples - n_sampled)
            samples[n_sampled: n_sampled + size] = samples_batch[:size]
            n_sampled += size

        print("total number of sampled features: {}".format(n_sampled))
        samples = samples[:n_sampled]
        self.in_head.unsup_train(samples, init=init)

        for i, layer in enumerate(self.layers):
            print("Training layer {}".format(i + 1))
            n_sampled = 0
            samples = torch.Tensor(n_samples, layer.input_size).to(device)

            for data in data_loader:
                data = data.to(device)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                deg_sqrt = getattr(data, 'deg_sqrt', None)
                x = self.representation(i, x, edge_index, edge_attr, deg_sqrt)
                samples_batch = layer.sample(x, edge_index, n_samples_per_batch)
                size = min(samples_batch.shape[0], n_samples - n_sampled)
                samples[n_sampled: n_sampled + size] = samples_batch[:size]
                n_sampled += size

            print("total number of sampled features: {}".format(n_sampled))
            samples = samples[:n_sampled]
            layer.unsup_train(samples, init=init)

            if layer.out_proj is not None:
                n_sampled = 0
                samples = torch.Tensor(n_samples, layer.out_proj.input_size).to(device)

                for data in data_loader:
                    data = data.to(device)
                    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                    deg_sqrt = getattr(data, 'deg_sqrt', None)
                    x = self.representation(i, x, edge_index, edge_attr,
                                            deg_sqrt, before_out_proj=True)
                    samples_batch = layer.out_proj.sample(x, n_samples_per_batch)
                    size = min(samples_batch.shape[0], n_samples - n_sampled)
                    samples[n_sampled: n_sampled + size] = samples_batch[:size]
                    n_sampled += size

                print("total number of sampled features: {}".format(n_sampled))
                samples = samples[:n_sampled]
                layer.out_proj.unsup_train(samples, init=init)






def load_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised Fisher information embedding for citation datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='name of dataset')

    # Model hyperparameters
    parser.add_argument('--num-layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of filters for layers')
    parser.add_argument('--num-mixtures', type=int, default=4,
                        help='number of mixtures for FIE layers')
    parser.add_argument('--num-heads', type=int, default=1,
                        help='number of heads for FIE layers')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma for Gaussian kernel layers')
    parser.add_argument('--concat', action='store_true',
                        help='concatenating features across layers')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='epochs')

    # Others
    parser.add_argument('--outdir', type=str, default=None,
                        help='output directory')

    args = parser.parse_args()

    args.device = torch.device(torch.cuda.current_device()) \
        if torch.cuda.is_available() else torch.device('cpu')

    if args.outdir is not None:
        args.outdir = args.outdir + \
            f'/{args.dataset}' + \
            f'/{args.num_layers}_{args.hidden_size}_{args.num_mixtures}_{args.sigma}_{args.concat}' + \
            f'/seed{args.seed}'
        os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args


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



def main():
    global args
    args = load_args()
    print(args)

    dataset = datasets.Planetoid(
        '../datasets/citation', name=args.dataset, split='public')

    node_features = dataset.data.x.numpy()
    # sc = StandardScaler().fit(node_features)

    def normalize_features(data):
        x_trans = data.x.numpy()
        x_trans /= np.linalg.norm(x_trans, axis=-1, keepdims=True).clip(min=1e-06)
        data.x = torch.from_numpy(x_trans)
        edge_index, edge_attr = utils.add_self_loops(
            data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data
    dataset.transform = T.Compose([normalize_features])

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    input_size = dataset.num_node_features
    dset = dataset[0]
    # train_mask = dset.train_mask.numpy()
    # val_mask = dset.val_mask.numpy()
    # test_mask = dset.test_mask.numpy()
    print(dset)
    dset = dset.to(args.device)

    model = SupFIENet(
        dataset.num_classes,
        input_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_mixtures=args.num_mixtures,
        num_heads=args.num_heads,
        out_proj_args=args.sigma,
        pooling=None,
        concat=True
    )
    model.to(args.device)
    print(model)

    model.unsup_train(data_loader, n_samples=300000)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-4)

    best_val_acc = 0
    best_epoch = 0
    tic = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(
            epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_epoch(dset, model, optimizer)
        val_loss, val_acc = eval_epoch(dset, model, split='Val')
        lr_scheduler.step(val_acc)
        eval_epoch(dset, model, split='Test')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    run_time = timer() - tic
    model.load_state_dict(best_weights)

    output = model(dset)
    output = output.cpu().detach().numpy()
    print(output.shape)

    node_labels = dset.y.cpu().detach().numpy()
    num_classes = len(set(node_labels))
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(output)


    fig = plt.figure()
    for class_id in range(num_classes):
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20)

    #plt.legend()
    #plt.show()
    plt.title("FIE Supervised")
    plt.xticks([],[])
    plt.yticks([],[])
    fig.tight_layout()
    fig.savefig("tsne_fie_sup.pdf")


if __name__ == '__main__':
    main()
