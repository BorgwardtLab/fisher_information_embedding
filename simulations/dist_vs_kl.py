import math
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric import utils
from torch_scatter import scatter
from einops import rearrange, repeat
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize" : (7, 4),
    "font.size": 18
})

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def gmm_attn2(x, mu0, log_pi):
    # x = n * d
    p, h, d = mu0.shape
    n = x.shape[0]
    mu0 = rearrange(mu0, "p h d -> (p h) d")
    log_pi = rearrange(log_pi, "p h -> (p h)")
    x = x.unsqueeze(1).expand(n, p*h, d)
    mu0 = mu0.unsqueeze(0).expand(n, p*h, d)
    dist = torch.pow(x - mu0, 2).sum(2)
    dist = rearrange(dist, "n (p h) -> n p h", p=p) # n * p * h
    attn = F.softmax(-dist, dim=1) # doing softmax is attn_j = exp(-|x-mu0_j|^2)/sum_l exp(-|x-mu0_l|^2)
    return attn # n * p * h


class FIEEmbedder(nn.Module):
    def __init__(self, input_size, num_mixtures, num_em_iters=1):
        super().__init__()
        self.input_size = input_size
        self.num_mixtures = num_mixtures
        self.num_em_iters = num_em_iters

        self.mu0 = nn.Parameter(
            torch.Tensor(num_mixtures, 1, input_size))
        self.bias = nn.Parameter(
            torch.zeros(num_mixtures, 1)
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for w in self.parameters():
            if w.ndim > 2:
                nn.init.normal_(w.data)
                w.data = F.normalize(w.data, dim=-1)

    @torch.no_grad()
    def normalize_(self):
        self.mu0.data.copy_(F.normalize(self.mu0.data, dim=-1))

    @torch.no_grad()
    def unsup_train_rand(self, x):
        #x : n * d
        for i in range(self.num_mixtures):
            self.mu0[i] = x[np.random.randint(x.shape[0])]

    @torch.no_grad()
    def forward(self, x):
        # x = n * d
        mu = self.mu0 # p * h * d
        for _ in range(self.num_em_iters):
            alpha = gmm_attn2(x, mu, self.bias) # n * num_mixtures * 1
            alpha = rearrange(alpha, "n p h -> h n p") # h * n * num_mixtures
            alpha = torch.clamp(alpha, min=1e-30)
            alpha = alpha/alpha.sum(1)

            mu = torch.matmul(torch.transpose(alpha, 1, 2), x) # h * p * d
            mu = rearrange(mu, "h p d -> p h d") 

            #print("mu", mu)
        return mu #  p * h * d



def run_exp(n_exp):
    de = 0
    if n_exp == 0: # fix number of components in one distr. = 3 and increase comp. k in other distr
        n_samples = 5000
        n_features = 2
        num_mixtures_est = 10


        maxk = 10
        n_iters = 5

        range_k = range(1, maxk+1)
        out = [np.zeros(n_iters) for _ in range_k]

        for it in range(n_iters):
            fie = FIEEmbedder(n_features, num_mixtures_est, num_em_iters=10)
            for k in range_k:
                mu1 = [[-5, -2]+[0 for _ in range(n_features-2)],
                    [-5-de, 0]+[0 for _ in range(n_features-2)],
                    [-5, 2]+[0 for _ in range(n_features-2)]
                    ]

                mu2 = [ [5, x]+[0 for _ in range(n_features-2)] for x in np.linspace(-2, 2, num=k)]

                goldberg_kl = 0
                for mu2j in mu2:
                    goldberg_kl += min([ np.linalg.norm((np.array(mu2j) - np.array(mu1j)))**2 for mu1j in mu1])/(2.0*len(mu2))
                
                var = 1
                cov = np.diag([var for _ in range(n_features)])

                distrib1 = [torch.from_numpy(np.random.multivariate_normal(mu1[i], cov)).float() for i in np.random.choice([0,1,2], size=n_samples)]
                distrib2 = [torch.from_numpy(np.random.multivariate_normal(mu2[i], cov)).float() for i in np.random.choice(range(k), size=n_samples)]
                distrib1 = rearrange(distrib1, "n d -> n d")
                distrib2 = rearrange(distrib2, "n d -> n d")

                distrib_tot = torch.cat([distrib1, distrib2], dim = 0)
                if unsup_train:
                    fie.unsup_train_rand(distrib_tot)
                phi1 = fie.forward(distrib1)/math.sqrt(num_mixtures_est)
                phi2 = fie.forward(distrib2)/math.sqrt(num_mixtures_est)

                out[k-range_k[0]][it] = (((phi1-phi2).norm()**2) / (2*goldberg_kl)).detach().numpy()

        mean = np.zeros(len(range_k))
        std = np.zeros(len(range_k))
        for k in range_k:
            mean[k-range_k[0]] = np.mean(out[k-range_k[0]])
            std[k-range_k[0]] = np.std(out[k-range_k[0]])

        print(mean)
        print(std)
        print("")
        plt.plot(range_k, mean, '-o')
        plt.fill_between(range_k, mean-std, mean+std, alpha=0.25)
        plt.xticks(range_k, range_k)
        #plt.title(r"Vary n. of components in target distrubution")
        plt.xlabel(r"n. of components in underlying distribution $\kappa$", fontsize=22)
        plt.ylabel(r" $ \|\phi(p_1) - \phi(p_2)\|^2 / 2D(p_1 \| p_2)$", fontsize=18)
        plt.ylim([0.8, 1.2])


    if n_exp == 1: # fix number of components in both distr., vary distance between distribs
        n_samples = 5000
        n_features = 2
        num_mixtures_est = 10


        n_iters = 20

        range_d = [1, 3.1, 10, 31, 100]
        out = [np.zeros(n_iters) for _ in range_d]

        for it in range(n_iters):
            fie = FIEEmbedder(n_features, num_mixtures_est, num_em_iters=10)
            for i in range(len(range_d)):
                d = range_d[i]
                mu1 = [[-5, -2]+[0 for _ in range(n_features-2)],
                    [-5-de, 0]+[0 for _ in range(n_features-2)],
                    [-5, 2]+[0 for _ in range(n_features-2)]
                    ]

                mu2 = [[-5+d, -2]+[0 for _ in range(n_features-2)],
                    [-5+d+de, 0]+[0 for _ in range(n_features-2)],
                    [-5+d, 2]+[0 for _ in range(n_features-2)]
                    ]

                goldberg_kl = 0
                for mu2j in mu2:
                    goldberg_kl += min([ np.linalg.norm((np.array(mu2j) - np.array(mu1j)))**2 for mu1j in mu1])/(2.0*len(mu2))
                
                var = 1
                cov = np.diag([var for _ in range(n_features)])

                distrib1 = [torch.from_numpy(np.random.multivariate_normal(mu1[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib2 = [torch.from_numpy(np.random.multivariate_normal(mu2[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib1 = rearrange(distrib1, "n d -> n d")
                distrib2 = rearrange(distrib2, "n d -> n d")

                distrib_tot = torch.cat([distrib1, distrib2], dim = 0)
                if unsup_train:
                    fie.unsup_train_rand(distrib_tot)
                phi1 = fie.forward(distrib1)/math.sqrt(num_mixtures_est)
                phi2 = fie.forward(distrib2)/math.sqrt(num_mixtures_est)

                out[i][it] = (((phi1-phi2).norm()**2) / (2*goldberg_kl)).detach().numpy()

        mean = np.zeros(len(range_d))
        std = np.zeros(len(range_d))
        for i in range(len(range_d)):
            mean[i] = np.mean(out[i])
            std[i] = np.std(out[i])

        print(mean)
        print(std)
        print("")
        plt.semilogx(range_d, mean, '-o')
        plt.fill_between(range_d, mean-std, mean+std, alpha=0.25)
        #plt.title(r"Vary distance between averages of the two distrubutions")
        plt.xlabel(r"Distance between averages $d$", fontsize=25)
        plt.ylabel(r" $ \|\phi(p_1) - \phi(p_2)\|^2 / 2D(p_1 \| p_2)$", fontsize=18)
        plt.ylim([0.8, 1.2])


    if n_exp == 2: # fix number of components in both distr., vary distance between components
        n_samples = 5000
        n_features = 2
        num_mixtures_est = 10


        n_iters = 20

        range_d = [0.1, 0.31, 1, 3.1, 10]
        out = [np.zeros(n_iters) for _ in range_d]

        for it in range(n_iters):
            for i in range(len(range_d)):
                fie = FIEEmbedder(n_features, num_mixtures_est, num_em_iters=10)
                d = range_d[i]
                mu1 = [[-5, -d]+[0 for _ in range(n_features-2)],
                    [-5-de, 0]+[0 for _ in range(n_features-2)],
                    [-5, d]+[0 for _ in range(n_features-2)]
                    ]

                mu2 = [[5, -d]+[0 for _ in range(n_features-2)],
                    [5+de, 0]+[0 for _ in range(n_features-2)],
                    [5, d]+[0 for _ in range(n_features-2)]
                    ]

                goldberg_kl = 0
                for mu2j in mu2:
                    goldberg_kl += min([ np.linalg.norm((np.array(mu2j) - np.array(mu1j)))**2 for mu1j in mu1])/(2.0*len(mu2))
                
                var = 1
                cov = np.diag([var for _ in range(n_features)])

                distrib1 = [torch.from_numpy(np.random.multivariate_normal(mu1[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib2 = [torch.from_numpy(np.random.multivariate_normal(mu2[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib1 = rearrange(distrib1, "n d -> n d")
                distrib2 = rearrange(distrib2, "n d -> n d")

                distrib_tot = torch.cat([distrib1, distrib2], dim = 0)
                if unsup_train:
                    fie.unsup_train_rand(distrib_tot)
                phi1 = fie.forward(distrib1)/math.sqrt(num_mixtures_est)
                phi2 = fie.forward(distrib2)/math.sqrt(num_mixtures_est)

                out[i][it] = (((phi1-phi2).norm()**2) / (2*goldberg_kl)).detach().numpy()

        mean = np.zeros(len(range_d))
        std = np.zeros(len(range_d))
        for i in range(len(range_d)):
            mean[i] = np.mean(out[i])
            std[i] = np.std(out[i])

        print(mean)
        print(std)
        print("")
        plt.semilogx(range_d, mean, '-o')
        plt.fill_between(range_d, mean-std, mean+std, alpha=0.25)
        #plt.title(r"Vary distance between components in each distribution")
        plt.xlabel(r"Distance between components $d$", fontsize=25)
        plt.ylabel(r" $ \|\phi(p_1) - \phi(p_2)\|^2 / 2D(p_1 \| p_2)$", fontsize=18)
        plt.ylim([0.8, 1.2])




    if n_exp == 3: # vary n. em iterations
        n_samples = 5000
        n_features = 2
        num_mixtures_est = 10


        n_iters = 20

        range_v = [0, 1, 2, 4, 6, 8, 10]
        out = [np.zeros(n_iters) for _ in range_v]

        for it in range(n_iters):
            for i in range(len(range_v)):
                fie = FIEEmbedder(n_features, num_mixtures_est, num_em_iters=range_v[i])
                mu1 = [[-5, -2]+[0 for _ in range(n_features-2)],
                    [-5-de, 0]+[0 for _ in range(n_features-2)],
                    [-5, 2]+[0 for _ in range(n_features-2)]
                    ]

                mu2 = [[5, -2]+[0 for _ in range(n_features-2)],
                    [5+de, 0]+[0 for _ in range(n_features-2)],
                    [5, 2]+[0 for _ in range(n_features-2)]
                    ]


                var = 1
                cov = np.diag([var for _ in range(n_features)])

                goldberg_kl = 0
                for mu2j in mu2:
                    goldberg_kl += min([ np.linalg.norm((np.array(mu2j) - np.array(mu1j)))**2 for mu1j in mu1])/(2.0*len(mu2)*var)
                
                distrib1 = [torch.from_numpy(np.random.multivariate_normal(mu1[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib2 = [torch.from_numpy(np.random.multivariate_normal(mu2[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib1 = rearrange(distrib1, "n d -> n d")
                distrib2 = rearrange(distrib2, "n d -> n d")

                distrib_tot = torch.cat([distrib1, distrib2], dim = 0)
                if unsup_train:
                    fie.unsup_train_rand(distrib_tot)
                phi1 = fie.forward(distrib1)/math.sqrt(num_mixtures_est)
                phi2 = fie.forward(distrib2)/math.sqrt(num_mixtures_est)

                out[i][it] = (((phi1-phi2).norm()**2) / (2*goldberg_kl)).detach().numpy()

        mean = np.zeros(len(range_v))
        std = np.zeros(len(range_v))
        for i in range(len(range_v)):
            mean[i] = np.mean(out[i])
            std[i] = np.std(out[i])

        print(mean)
        print(std)
        print("")
        plt.plot(range_v, mean, '-o')
        plt.fill_between(range_v, mean-std, mean+std, alpha=0.25)
        #plt.title(r"Vary umber of EM iterations")
        plt.xlabel(r"EM iterations $T$", fontsize=25)
        plt.ylabel(r" $ \|\phi(p_1) - \phi(p_2)\|^2 / 2D(p_1 \| p_2)$", fontsize=18)
        plt.ylim([0, 1.2])



    if n_exp == 4: # vary n. mixtures
        n_samples = 5000
        n_features = 2
        de = 4

        n_iters = 20

        range_v = [1, 3, 10, 30, 100]
        out = [np.zeros(n_iters) for _ in range_v]

        for it in range(n_iters):
            for i in range(len(range_v)):
                num_mixtures_est = range_v[i]
                fie = FIEEmbedder(n_features, num_mixtures_est, num_em_iters=10)
                mu1 = [[-5, -2]+[0 for _ in range(n_features-2)],
                    [-5-de, 0]+[0 for _ in range(n_features-2)],
                    [-5, 2]+[0 for _ in range(n_features-2)]
                    ]

                mu2 = [[5, -2]+[0 for _ in range(n_features-2)],
                    [5+de, 0]+[0 for _ in range(n_features-2)],
                    [5, 2]+[0 for _ in range(n_features-2)]
                    ]


                var = 1
                cov = np.diag([var for _ in range(n_features)])

                goldberg_kl = 0
                for mu2j in mu2:
                    goldberg_kl += min([ np.linalg.norm((np.array(mu2j) - np.array(mu1j)))**2 for mu1j in mu1])/(2.0*len(mu2)*var)
                
                distrib1 = [torch.from_numpy(np.random.multivariate_normal(mu1[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib2 = [torch.from_numpy(np.random.multivariate_normal(mu2[i], cov)).float() for i in np.random.choice(range(3), size=n_samples)]
                distrib1 = rearrange(distrib1, "n d -> n d")
                distrib2 = rearrange(distrib2, "n d -> n d")

                distrib_tot = torch.cat([distrib1, distrib2], dim = 0)
                if unsup_train:
                    fie.unsup_train_rand(distrib_tot)
                phi1 = fie.forward(distrib1)/math.sqrt(num_mixtures_est)
                phi2 = fie.forward(distrib2)/math.sqrt(num_mixtures_est)

                out[i][it] = (((phi1-phi2).norm()**2) / (2*goldberg_kl)).detach().numpy()

        mean = np.zeros(len(range_v))
        std = np.zeros(len(range_v))
        for i in range(len(range_v)):
            mean[i] = np.mean(out[i])
            std[i] = np.std(out[i])

        print(mean)
        print(std)
        print("")
        plt.semilogx(range_v, mean, '-o')
        plt.fill_between(range_v, mean-std, mean+std, alpha=0.25)
        #plt.title(r"Vary number of mixtures")
        plt.xlabel(r"Number of mixtures $k$", fontsize=25)
        plt.ylabel(r" $ \|\phi(p_1) - \phi(p_2)\|^2 / 2D(p_1 \| p_2)$", fontsize=18)
        plt.ylim([0.8, 1.4])

unsup_train = False
if len(sys.argv) > 1:
    n_exp = int(sys.argv[1])
    run_exp(n_exp)
    plt.tight_layout()
    plt.savefig("exp_"+str(n_exp)+".png", dpi=600)
    plt.show()
else:
    for n_exp in range(5):
        run_exp(n_exp)

        plt.tight_layout()
        plt.savefig("exp_"+str(n_exp)+".png", dpi=600)
        plt.show()
