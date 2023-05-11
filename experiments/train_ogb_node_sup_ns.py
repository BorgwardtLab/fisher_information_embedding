# -*- coding: utf-8 -*-
import os
import copy
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader, NeighborSampler
import torch_geometric.transforms as T

import pandas as pd
import argparse

from timeit import default_timer as timer
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from fie.models import ReLUFIENet


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
    parser.add_argument('--no-concat', action='store_true',
                        help='do not concatenate features across layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='epochs')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers for loaders')

    parser.add_argument('--outdir', type=str, default=None,
                        help='output directory')

    args = parser.parse_args()

    args.device = torch.device(torch.cuda.current_device()) \
        if torch.cuda.is_available() else torch.device('cpu')
    args.concat = not args.no_concat

    if args.outdir is not None:
        args.outdir = args.outdir + \
            f'/{args.dataset}' + \
            f'/{args.num_layers}_{args.hidden_size}_{args.num_mixtures}_{args.dropout}_{args.concat}' + \
            f'/seed{args.seed}'
        os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args


def train(model, data, train_loader, optimizer):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # data = data.to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]

        optimizer.zero_grad()
        out = model.forward_ns(data.x[n_id], adjs, batch_size)
        loss = F.cross_entropy(out, data.y[n_id[:batch_size]].view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # total_correct += (out.data.argmax(dim=-1) == data.y[n_id[:batch_size]]).sum().item()

    loss = total_loss / len(train_loader)
    # approx_acc = total_correct / train_loader.node_idx.size(0)
    # print(f"Train loss: {loss:.2f} Acc: {approx_acc:.2f}")

    return loss


@torch.no_grad()
def test(model, data, test_loader, split_idx, evaluator):
    model.eval()

    out = model.inference_ns(data.x, test_loader)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    global args
    args = load_args()
    print(args)

    dataset = PygNodePropPredDataset(
        root='../datasets/ogbn', name=args.dataset)

    # node_features = dataset.data.x.numpy()
    # sc = StandardScaler().fit(node_features)

    def preprocess(data):
        # x_trans = data.x.numpy()
        # x_trans /= np.linalg.norm(x_trans, axis=-1, keepdims=True).clip(min=1e-06)
        # data.x = torch.from_numpy(x_trans)
        adj = data.adj_t.to_symmetric()
        row, col, _ = adj.coo()
        edge_index = torch.stack((row, col))
        data.adj_t = None
        edge_index, _ = utils.add_self_loops(
            edge_index, None, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        # deg_log = utils.degree(data.edge_index[0], data.num_nodes, dtype=data.x.dtype)
        # deg_log = (deg_log + 1.).log()
        # deg_log = (deg_log - deg_log.mean()) / deg_log.std()
        # data.x = torch.cat((data.x, deg_log.view(-1, 1)), dim=-1)
        return data
    dataset.transform = T.Compose([T.ToSparseTensor(), preprocess])

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    input_size = dataset.num_node_features
    data = dataset[0]
    data = data.to(args.device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(args.device)
    print(data)
    print(data.x)
    print(train_idx)

    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                   sizes=[10] * args.num_layers, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)

    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=2 * args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    model = ReLUFIENet(
        dataset.num_classes,
        input_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_mixtures=args.num_mixtures,
        num_heads=args.num_heads,
        concat=args.concat,
        dropout=args.dropout
    )
    model.to(args.device)

    # model.unsup_train(data_loader, n_samples=300000)

    evaluator = Evaluator(name=args.dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    best_epoch = 0
    run_time = 0
    for epoch in range(1, 1 + args.epochs):
        tic = timer()
        loss = train(model, data, train_loader, optimizer)
        train_time = timer() - tic
        run_time += train_time
        if epoch > args.epochs - 50 or args.dataset == 'ogbn-arxiv':
            tic = timer()
            result = test(model, data, subgraph_loader, split_idx, evaluator)
            val_time = timer() - tic
            train_acc, valid_acc, test_acc = result

            if valid_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = valid_acc
                best_weights = copy.deepcopy(model.state_dict())

            print(
                  f'Epoch: {epoch:02d}/{args.epochs}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}% ({train_time:.2f}s), '
                  f'Valid: {100 * valid_acc:.2f}% ({val_time:.2f}s) '
                  f'Test: {100 * test_acc:.2f}%')
        else:
            print(
                  f'Epoch: {epoch:02d}/{args.epochs}, '
                  f'Loss: {loss:.4f}, '
                  f'Time: {train_time:.2f}s')

    model.load_state_dict(best_weights)

    train_acc, val_acc, test_acc = test(model, data, subgraph_loader, split_idx, evaluator)

    print("Best val epoch: {} val Acc {:.4f} test Acc {:.4f}".format(
        best_epoch, val_acc, test_acc))

    if args.outdir is not None:
        results = [{
            'test_score': test_acc,
            'val_score': val_acc,
            'train_score': train_acc,
            'best_epoch': best_epoch,
            'time': run_time,
        }]
        pd.DataFrame(results).to_csv(f'{args.outdir}/results.csv')


if __name__ == '__main__':
    main()
