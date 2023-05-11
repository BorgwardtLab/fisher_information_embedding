# Fisher information embedding for node and graph learning

This repository implements the Fisher information embedding (FIE) described in the following paper

>Dexiong Chen*, Paolo Pellizzoni*, and Karsten Borgwardt.
[Fisher Information Embedding for Node and Graph Learning][1]. ICML 2023.
<br/>*Equal contribution

**TL;DR**: a class of node embeddings with an information geometry interpretation, available with both unsupervised and supervised algorithms for learning the embeddings.


## Citation

Please use the following to cite our work:

```bibtex
@InProceedings{Chen23fie,
    author = {Dexiong Chen and Paolo Pellizzoni and Karsten Borgwardt},
    title = {Fisher Information Embedding for Node and Graph Learning},
    year = {2023},
    booktitle = {International Conference on Machine Learning~(ICML)},
    series = {Proceedings of Machine Learning Research}
}
```

## A short description of FIE

TODO Paolo: figure + some texts

#### Quickstart

<details><summary>click to see the example</summary>

```python
from torch_geometric import datasets
from torch_geometric.loader import DataLoader

# Construct data loader
dataset = datasets.Planetoid('./datasets/citation', name='Cora', split='public')
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
input_size = dataset.num_node_features

# Build FIE model
model = FIENet(
    input_size,
    num_layers=2,
    hidden_size=16,
    num_mixtures=8,
    pooling=None,
    concat=True
)

# Train model parameters using k-means
model.unsup_train(data_loader)

# Compute node embeddings
X = model.predict(data_loader)
```
</details>

## Installation

The dependencies are managed by [miniconda][2]. Run the following to install the dependencies

```bash
# For CPU only
conda env create -f env.yml
# Or if you have a GPU
conda env create -f env_cuda.yml
# Then activate the environment
conda activate fie
```

Then, install our `fisher_information_embedding` package:

```bash
pip install -e .
```

## Training models

Please see Table 3 and 4 in our paper to find the search grids for each hyperparameter. Note that we use very minimal hyperparameter tuning in our paper.

#### Training models on semi-supervised learning tasks using citation networks

- Unsupervised node embedding mode with logistic classifier:
  ```bash
  python train_citation.py --dataset Cora --hidden-size 512 --num-mixtures 8 --num-layers 4
  ```
- Supervised node embedding mode:
  ```bash
  python train_citation_sup.py --dataset Cora --hidden-size 64 --num-mixtures 8 --num-layers 4
  ```

#### Training models on supervised learning tasks using large OGB datasets

- Unsupervised node embedding mode with FLAML:
  ```bash
  python train_ogb_node.py --save-memory --dataset ogbn-arxiv --hidden-size 256 --num-mixtures 8 --num-layers 5
  ```
- Supervised node embedding mode:
  ```bash
  python train_ogb_node_sup_ns.py --dataset ogbn-arxiv --hidden-size 256 --num-mixtures 4 --num-layers 3
  ```



[1]: TODO
[2]: https://conda.io/miniconda.html
