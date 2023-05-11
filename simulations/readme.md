# Simulations
These are some simulations to assess if the distance in the embedding space closely matches the KL divergence between two distributions $p_1$ and $p_2$.

In the first set of experiments, we keep the parameters of the embedding method fixed and we vary the underlying distributions, to understand in which cases the fisher embedding approximates well the KL divergence. 
In particular, we use as the family of distributions a family of Gaussian mixtures with $k =10$ components, and with each component having identity covariance. 
Moreover, we fix the number of EM iterations to find the ML estimator to $T=10$. 

We then vary the underlying data-generating distributions as follows.

 - In the first simulation we have $p_1$ as a mixture of three
  Gaussians $\mathcal{N}(\mu_{1,j}, I)$, with
  $\mu_{1,1} = (-5, -2)$, $\mu_{1,1} = (-5, 0)$,
  $\mu_{1,1} = (-5, 2)$. Then we vary $p_2$ as a mixture of $\kappa$
  Gaussians $\mathcal{N}(\mu_{2,j}, I)$, with means evenly spaced
  between $(5,-2)$ and $(5, 2)$, for $\kappa \in [1, 10]$.
-  In the second simulation, we have that both the distributions are
  mixtures of three Gaussians $\mathcal{N}(\mu_{i,j}, I)$, and we vary
  the distance between the means of the two distributions, keeping the
  distance between components of the same distribution fixed. In
  particular, we let $\mu_{1,1} = (-5, -2)$, $\mu_{1,1} = (-5, 0)$,
  $\mu_{1,1} = (-5, 2)$ and $\mu_{2,1} = (-5+d, -2)$,
  $\mu_{2,1} = (-5+d, 0)$, $\mu_{2,1} = (-5+d, 2)$, for
  $d \in [1, 100]$.
-  In the third simulation, we again have that both the distributions are
  mixtures of three Gaussians $\mathcal{N}(\mu_{i,j}, I)$, but this
  time we vary the distance between the components of each distribution,
  keeping the distance between the means of the two distributions fixed.
  In paricular, we let $\mu_{1,1} = (-5, -d)$,
  $\mu_{1,1} = (-5, 0)$, $\mu_{1,1} = (-5, d)$ and
  $\mu_{2,1} = (5, -d)$, $\mu_{2,1} = (5, 0)$,
  $\mu_{2,1} = (5, d)$, for $d \in [0.1, 10]$.

In the second set of experiments we keep the underlying distributions fixed and the vary the parameters of our embedding method. 
In particular, we let both distributions be a
  mixture of three Gaussians $\mathcal{N}(\mu_{1,j}, I)$ 
  with $\mu_{1,1} = (-5, -2)$,
  $\mu_{1,1} = (-5, 0)$, $\mu_{1,1} = (-5, 2)$ and
  $\mu_{2,1} = (5, -2)$, $\mu_{2,1} = (5, 0)$,
  $\mu_{2,1} = (5, 2)$.

We then vary the number $k$ of components in the parametric family of Gaussian mixtures for the maximum likelihood estimation in $[1, 100]$, and the number $T$ of EM iterations in $[0,10]$. 