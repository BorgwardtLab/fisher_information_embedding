# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from .layers import FIELayer, FIEPooling, KernelLayer


class FIENet(nn.Module):
    def __init__(self, input_size, num_layers=2, hidden_size=64, num_mixtures=1, num_heads=1,
                 residue=True, out_proj='kernel', out_proj_args=1.0, use_deg=False,
                 pooling='fie', concat=False):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.concat = concat

        self.in_head = KernelLayer(input_size, hidden_size, sigma=out_proj_args)

        layers = []

        for i in range(num_layers - 1):
            layers.append(
                FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
                         residue, out_proj, 'exp', out_proj_args, use_deg)
            )

        layers.append(
            FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
                     residue, None, 'exp', out_proj_args, use_deg)
        )

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

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        deg_sqrt = getattr(data, 'deg_sqrt', None)

        x = self.in_head(x)

        outputs = [x]
        output = x
        for i, mod in enumerate(self.layers):
            # if i == self.num_layers - 1:
            #     output = mod(outputs[-1], edge_index, edge_attr, deg_sqrt=deg_sqrt, before_out_proj=True)
            # else:
            #     output = mod(output, edge_index, edge_attr, deg_sqrt=deg_sqrt)
            output = mod(output, edge_index, edge_attr, deg_sqrt=deg_sqrt)
            # output = mod(output, edge_index, edge_attr, deg_sqrt=deg_sqrt, before_out_proj=True)
            outputs.append(output)
            # output = mod.out_proj(output) * deg_sqrt.view(-1, 1)

        if self.concat:
            output = torch.cat(outputs, dim=-1)
        else:
            output = outputs[-1]

        if self.pooling is not None:
            output = self.pooling(output, data.batch)
        return output

    @torch.no_grad()
    def predict(self, data_loader, device='cpu'):
        self.eval()

        outputs = []
        for data in data_loader:
            data = data.to(device)
            outputs.append(self(data).cpu())

        outputs = torch.cat(outputs)
        return outputs

    @torch.no_grad()
    def predict_ns(self, data, data_loader):
        x = data.x
        deg_sqrt = getattr(data, 'deg_sqrt', None)
        device = x.device
        print(f"device: {device}")
        x = self.in_head(x)

        outputs = [x]
        for i, mod in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in data_loader:
                edge_index, _, size = adj.to(device)
                x_batch = x[n_id].to(device)
                x_target = x_batch[:size[1]]
                x_batch = mod((x_batch, x_target), edge_index, None, deg_sqrt=deg_sqrt)
                xs.append(x_batch.cpu())
            x = torch.cat(xs, dim=0)
            outputs.append(x)

        if self.concat:
            x = torch.cat(outputs, dim=-1)
        return x

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
    def unsup_train(self, data_loader, n_samples=300000,
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

    @torch.no_grad()
    def unsup_train_ns(self, data, data_loader, n_samples=100000,
                       init=None):
        self.train(False)
        x = data.x
        edge_index = data.edge_index
        deg_sqrt = getattr(data, 'deg_sqrt', None)
        device = x.device
        try:
            n_samples_per_batch = (n_samples + len(data_loader) - 1) // len(data_loader)
        except Exception:
            n_samples_per_batch = 1000
 
        samples = self.in_head.sample(x, n_samples)
        print("total number of sampled features: {}".format(samples.shape[0]))
        self.in_head.unsup_train(samples, init=init)

        x = self.in_head(x)
        outputs = [x]
        for i, mod in enumerate(self.layers):
            print("-" * 20)
            print(f"Training layer {i + 1}")
            samples = mod.sample(x, edge_index, n_samples)
            print(f"sampled features shape: {samples.shape}")
            mod.unsup_train(samples, init=init)

            if mod.out_proj is not None:
                print("training projection layer")
                xs = []
                for batch_size, n_id, adj in data_loader:
                    edge_index_batch, _, size = adj.to(device)
                    x_batch = x[n_id].to(device)
                    x_target = x_batch[:size[1]]
                    x_batch = mod((x_batch, x_target), edge_index_batch, None, deg_sqrt=deg_sqrt,
                                  before_out_proj=True)
                    xs.append(x_batch.cpu())
                x = torch.cat(xs, dim=0)
                samples = mod.out_proj.sample(x, n_samples)
                print(f"sampled features shape: {samples.shape}")
                mod.out_proj.unsup_train(samples, init=init)
                x = mod.forward_proj(x, edge_index, deg_sqrt=deg_sqrt)
            else:
                xs = []
                for batch_size, n_id, adj in data_loader:
                    edge_index_batch, _, size = adj.to(device)
                    x_batch = x[n_id].to(device)
                    x_target = x_batch[:size[1]]
                    x_batch = mod((x_batch, x_target), edge_index_batch, None, deg_sqrt=deg_sqrt)
                    xs.append(x_batch.cpu())
                x = torch.cat(xs, dim=0)

            outputs.append(x)

        if self.concat:
            x = torch.cat(outputs, dim=-1)
        return x


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

        if self.concat:
            output = torch.cat(outputs, dim=-1)
        else:
            output = outputs[-1]

        if self.pooling is not None:
            output = self.pooling(output, data.batch)
        return self.classifier(output)

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


class ReLUFIENet(nn.Module):
    def __init__(self, num_class, input_size, num_layers=2,
                 hidden_size=64, num_mixtures=1, num_heads=1,
                 residue=True, out_proj='relu', use_deg=False,
                 dropout=0.0, concat=False):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.concat = concat
        self.dropout = dropout

        self.in_head = nn.Linear(input_size, hidden_size)

        layers = []

        for i in range(num_layers):
            layers.append(
                FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
                         residue, out_proj, 'exp', 0.5, use_deg)
            )
        self.layers = nn.ModuleList(layers)

        output_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(output_size, num_class))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        deg_sqrt = getattr(data, 'deg_sqrt', None)

        x = self.in_head(x)

        outputs = x
        output = x
        for i, mod in enumerate(self.layers):
            output = mod(output, edge_index, edge_attr, deg_sqrt=deg_sqrt)
            if self.concat:
                output = outputs + output
                outputs = output
            if i < self.num_layers - 1:
                output = F.dropout(output, p=self.dropout, training=self.training)

        return self.classifier(output)

    def forward_ns(self, x, adjs, batch_size):
        x = self.in_head(x)

        outputs = x#[:batch_size]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.layers[i]((x, x_target), edge_index)
            if self.concat:
                x = x_target + x
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.classifier(x)

    def inference_ns(self, x_all, subgraph_loader):
        device = x_all.device
        x_all = self.in_head(x_all)
        # outputs = [x_all.cpu()]
        outputs = x_all.cpu()
        for i, mod in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = mod((x, x_target), edge_index)
                if self.concat:
                    x = x_target + x
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)

        return self.classifier(x_all.to(device))
