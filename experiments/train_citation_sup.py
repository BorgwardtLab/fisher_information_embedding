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

from fie.models import SupFIENet


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
    parser.add_argument('--hidden-size', type=int, default=16,
                        help='number of filters for layers')
    parser.add_argument('--num-mixtures', type=int, default=4,
                        help='number of mixtures for FIE layers')
    parser.add_argument('--num-heads', type=int, default=1,
                        help='number of heads for FIE layers')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma for Gaussian kernel layers')
    parser.add_argument('--no-concat', action='store_true',
                        help='do not concatenate features across layers')

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
    args.concat = not args.no_concat

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
        concat=args.concat
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

    test_loss, test_acc = eval_epoch(dset, model, split='Test')

    print(f"test Acc {test_acc:.4f} time: {run_time:.2f}s")

    if args.outdir is not None:
        results = [{
            'test_score': test_acc,
            'val_score': best_val_acc,
            'best_epoch': best_epoch,
            'time': run_time,
        }]
        pd.DataFrame(results).to_csv(args.outdir + '/results.csv')


if __name__ == '__main__':
    main()
