# Addressing Skewed Heterogeneity via Federated Prototype Rectification With Personalization

>  Shunxin Guo, Hongsong Wang, Shuxia Lin, Zhiqiang Kou, and Xin Geng, IEEE Transactions on Neural Networks and Learning Systems, 2024.
## Abstract
Federated learning (FL) is an efficient framework
designed to facilitate collaborative model training across
multiple distributed devices while preserving user data privacy.
A significant challenge of FL is data-level heterogeneity, i.e.,
skewed or long-tailed distribution of private data. Although
various methods have been proposed to address this challenge,
most of them assume that the underlying global data are
uniformly distributed across all clients. This article investigates
data-level heterogeneity FL with a brief review and redefines
a more practical and challenging setting called skewed
heterogeneous FL (SHFL). Accordingly, we propose a novel
federated prototype rectification with personalization (FedPRP)
which consists of two parts: federated personalization and
federated prototype rectification. The former aims to construct
balanced decision boundaries between dominant and minority
classes based on private data, while the latter exploits both
interclass discrimination and intraclass consistency to rectify
empirical prototypes. Experiments on three popular benchmarks
show that the proposed approach outperforms current state-of-the-art methods and achieves balanced performance in both
personalization and generalization.

```
    @article{guo2024addressing,
    title={Addressing Skewed Heterogeneity via Federated Prototype Rectification With Personalization},
    author={Guo, Shunxin and Wang, Hongsong and Lin, Shuxia and Kou, Zhiqiang and Geng, Xin},
    journal={IEEE Transactions on Neural Networks and Learning Systems},
    year={2024},
    publisher={IEEE}
    }
'''
