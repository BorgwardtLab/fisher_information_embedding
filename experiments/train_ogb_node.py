# -*- coding: utf-8 -*-
import os
import json
import random
import torch
import numpy as np
from torch_geometric import utils
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler

import pandas as pd
import argparse

from timeit import default_timer as timer
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from flaml.default import LGBMClassifier
from flaml import AutoML

from fie.models import FIENet


def load_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised Fisher information embedding for citation datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv",
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
    parser.add_argument('--budget', type=int, default=72000,
                        help="time budget for FLAML")
    parser.add_argument('--fast', action='store_true',
                        help='fast mode without k-means')
    parser.add_argument('--test', action='store_true',
                        help="test mode")
    parser.add_argument('--save-memory', action='store_true',
                        help='use neighbor sampler to save memory!')

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

    dataset = PygNodePropPredDataset(
        root='../datasets/ogbn', name=args.dataset)

    def preprocess(data):
        adj = data.adj_t.to_symmetric()
        row, col, _ = adj.coo()
        edge_index = torch.stack((row, col))
        data.adj_t = None
        edge_index, _ = utils.add_self_loops(
            edge_index, None, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        return data
    dataset.transform = T.Compose([T.ToSparseTensor(), preprocess])

    input_size = dataset.num_node_features
    dset = dataset[0]

    if args.save_memory:
        data_loader = NeighborSampler(dset.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=10000, shuffle=False,
                                      num_workers=0)
    else:
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].numpy()
    val_idx = split_idx['valid'].numpy()
    test_idx = split_idx['test'].numpy()

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

    tic = timer()
    X = None
    if not args.fast:
        if args.save_memory:
            X = model.unsup_train_ns(dset, data_loader, n_samples=300000)
        else:
            model.unsup_train(data_loader, n_samples=300000)
    unsup_time = timer() - tic
    print(f"Unsupervised training done, time: {unsup_time:.2f}s")

    tic = timer()
    if args.save_memory:
        if X is None:
            X = model.predict_ns(dset, data_loader)
    else:
        X = model.predict(data_loader, device=args.device)
    toc = timer()
    y = dset.y
    run_time = toc - tic

    print(f"Embedding finished, time: {run_time:.2f}s")

    print(X.shape)
    print(X)
    print(y)

    X, y = X.numpy(), y.numpy().ravel()
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    del X, y

    evaluator = Evaluator(name=args.dataset)

    automl = AutoML()
    settings = {
        "metric": 'accuracy',
        "task": 'classification',
        "estimator_list": ['lgbm'],
        "seed": args.seed,
        "time_budget": args.budget
    }

    if args.test:
        config_path = args.outdir.replace(f'seed{args.seed}', 'seed0')
        config_path = f'{config_path}/best_config.txt'
        with open(config_path, 'r') as f:
            best_config = json.load(f)

        if "log_max_bin" in best_config:
            best_config["max_bin"] = (1 << best_config.pop("log_max_bin")) - 1
        automl = LGBMClassifier(random_state=args.seed, **best_config)
        tic = timer()
        automl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                   eval_metric=['multi_error'])
        clf_time = timer() - tic
    else:
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val, **settings)
        clf_time = automl.best_config_train_time
    
    y_pred = automl.predict(X_te)
    score = evaluator.eval({
        'y_true': y_te.reshape(-1, 1),
        'y_pred': y_pred.reshape(-1, 1)
    })['acc']
    print(f"Test acc: {score * 100:.3f}%")
    val_score = evaluator.eval({
        'y_true': y_val.reshape(-1, 1),
        'y_pred': automl.predict(X_val).reshape(-1, 1)
    })['acc']

    if args.outdir is not None:
        results = [{
            'test_score': score,
            'val_score': val_score,
            # 'best_config': automl.best_config,
            'time': run_time,
            'clf_time': clf_time,
            'unsup_time': unsup_time
        }]
        suffix = '' if args.test else '_val'
        pd.DataFrame(results).to_csv(f'{args.outdir}/results{suffix}.csv')
        if not args.test:
            with open(f'{args.outdir}/best_config.txt', 'w') as f:
                f.write(json.dumps(automl.best_config))


if __name__ == '__main__':
    main()
