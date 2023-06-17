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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from timeit import default_timer as timer

from fie.models import FIENet

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize" : (10, 6),
    "font.size": 22
})


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
    parser.add_argument('--concat', action='store_true',
                        help='concatenating features across layers')
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
    print(dset)
    print(dset.x)

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


    X, y = X.numpy(), y.numpy()

    output = model(dset)
    output = output.cpu().detach().numpy()
    print(output.shape)

    node_labels = dset.y.cpu().detach().numpy()
    num_classes = len(set(node_labels))
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(output)


    fig = plt.figure()
    plt.title("FIE Unsupervised")
    for class_id in range(num_classes):
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20)

    #plt.legend()
    #plt.show()
    plt.xticks([],[])
    plt.yticks([],[])
    fig.tight_layout()
    fig.savefig("tsne_fie_unsup.pdf")


if __name__ == '__main__':
    main()
