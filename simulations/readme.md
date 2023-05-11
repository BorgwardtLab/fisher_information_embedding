# Simulations
These are some simulations to assess if the distance in the embedding space closely matches the KL divergence between two distributions $p_1$ and $p_2$.

- In the first simulation we have $p_1$ as a mixture of three gaussians $\mathcal{N}(\mu_{1,j}, I)$,
with $\mu_{1,1} = (10, -2)$, $\mu_{1,1} = (10, 0)$, $\mu_{1,1} = (10, 2)$.
Then we vary $p_2$ as a mixture of $k$ gaussians $\mathcal{N}(\mu_{2,j}, I)$, with means evenly spaced between $(20,-2)$ and $(20, 2)$, for $k \in [3, 10]$.
- In the second simulation, we have that both the distributions are mixtures of three gaussians $\mathcal{N}(\mu_{i,j}, I)$, and we vary the distance between the means of the two distributions,
keeping the distance between components of the same distribution fixed.
In particular, we let $\mu_{1,1} = (10, -2)$, $\mu_{1,1} = (10, 0)$, $\mu_{1,1} = (10, 2)$ and $\mu_{2,1} = (10+d, -2)$, $\mu_{2,1} = (10+d, 0)$, $\mu_{2,1} = (10+d, 2)$, for $d \in [10, 1000]$.
- In the third simulation, we again have that both the distributions are mixtures of three gaussians $\mathcal{N}(\mu_{i,j}, I)$, but this time we vary the distance between the components of each distribution,
keeping the distance between the means of the two distributions fixed.
In paricular, we let $\mu_{1,1} = (10, -d)$, $\mu_{1,1} = (10, 0)$, $\mu_{1,1} = (10, d)$ and $\mu_{2,1} = (20, -d)$, $\mu_{2,1} = (20, 0)$, $\mu_{2,1} = (20, d)$, for $d \in [0.1, 10]$.
- In the fourth simulation, we show that even if the covariance of the gaussians is not the identity, there is a linear dependence between the distance in the ambedding space and
the KL divergence betweenthe two distributions.
In particular, we let both distrbutions be a mixture of three gaussians $\mathcal{N}(\mu_{1,j}, \epsilon I)$, for $\epsilon \in [0.1, 10]$.
We let $\mu_{1,1} = (10, -2)$, $\mu_{1,1} = (10, 0)$, $\mu_{1,1} = (10, 2)$ and $\mu_{2,1} = (20, -2)$, $\mu_{2,1} = (20, 0)$, $\mu_{2,1} = (20, 2)$.
