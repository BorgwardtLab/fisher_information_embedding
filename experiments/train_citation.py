# -*- coding: utf-8 -*-
import os
import random
import torch
import numpy as np
from torch_geometric import datasets
from torch_geometric import utils
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from timeit import default_timer as timer

from fie.models import FIENet


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
    parser.add_argument('--num-mixtures', type=int, default=1,
                        help='number of mixtures for FIE layers')
    parser.add_argument('--num-heads', type=int, default=1,
                        help='number of heads for FIE layers')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma for Gaussian kernel layers')
    parser.add_argument('--no-concat', action='store_true',
                        help='do not concatenate features across layers')
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


def main():
    global args
    args = load_args()
    print(args)

    dataset = datasets.Planetoid(
        '../datasets/citation', name=args.dataset, split='public')

    node_features = dataset.data.x.numpy()
    sc = StandardScaler().fit(node_features)

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
    train_mask = dset.train_mask.numpy()
    val_mask = dset.val_mask.numpy()
    test_mask = dset.test_mask.numpy()

    model = FIENet(
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

    model.unsup_train(data_loader, n_samples=300000)

    tic = timer()
    X = model.predict(data_loader, device=args.device)
    toc = timer()
    y = dset.y

    run_time = toc - tic
    print(f"Embedding finished, time: {run_time:.2f}s")

    print(X.shape)
    print(X)
    print(y)

    X, y = X.numpy(), y.numpy()
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_tr_val, y_tr_val = np.vstack((X_tr, X_val)), np.hstack((y_tr, y_val))
    X_te, y_te = X[test_mask], y[test_mask]

    C_list = np.logspace(-4, 5, 19)
    scores = []
    tic = timer()
    for C in C_list:
        clf = LogisticRegression(C=C, max_iter=500)
        clf.fit(X_tr, y_tr)
        score = clf.score(X_val, y_val)
        scores.append(score)
        print("C: {} acc: {}".format(C, score))
    clf_time = timer() - tic

    val_score = np.max(scores)
    best_C = C_list[np.argmax(scores)]
    clf = LogisticRegression(C=best_C, max_iter=500)
    clf.fit(X_tr, y_tr)

    score = clf.score(X_te, y_te)
    print("Test acc: {:.3f}%".format(score * 100))
    train_score = clf.score(X_tr, y_tr)
    train_val_score = clf.score(X_tr_val, y_tr_val)

    if args.outdir is not None:
        results = [{
            'test_score': score,
            'val_score': val_score,
            'train_score': train_score,
            'train_val_score': train_val_score,
            'best_C': best_C,
            'time': run_time,
            'clf_time': clf_time
        }]
        pd.DataFrame(results).to_csv(args.outdir + '/results.csv')


if __name__ == '__main__':
    main()
