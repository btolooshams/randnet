"""
Copyright (c) 2020 Bahareh Tolooshams

config

:author: Thomas Change & Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "default",
        "dataset": "simulated",
        "network": "CRsAEDense",
        "seed": 103,
        "D_enc": 100,
        "D_comp": 50,
        "D_org": 100,
        "D_in": 100,
        "num_iters": 50,
        "L": 10,
        "lam": 0.1,
        "sparsity": 1,
        "randomness": 0.5,
        "num_phis": 10,
        "num_nonzero": 3,
        "num_samples": 100,
        "num_epochs": 2,
        "batch_size": 4,
        "test_batch": 64,
        "lr": 1e-3,
        "shuffle": True,
        "classification": False,
        "info_period": 10,
        "device": device,
    }


@config_ingredient.named_config
def crsae_simulated_test():
    hyp = {
        "experiment_name": "test_crsae_simulated",
        "dataset": "simulated",
        "network": "CRsAEDense",
        "seed": 103,
        "D_enc": 20,
        "D_comp": 50,
        "D_org": 20,
        "D_in": 100,
        "num_iters": 500,
        "L": 10,
        "lam": 1,
        "sparsity": 2,
        "randomness": 0.4,
        "num_phis": 10,
        "num_nonzero": 3,
        "num_samples": 100,
        "num_epochs": 30,
        "batch_size": 64,
        "lr": 3e-2,
        "shuffle": True,
        "classification": False,
        "info_period": 10000,
        "device": device,
    }


@config_ingredient.named_config
def crsae_mnist_test():
    hyp = {
        "experiment_name": "test_crsae_mnist",
        "dataset": "MNIST",
        "network": "CRsAERandProjAeClassifier",
        "seed": 103,
        "D_enc": 784,
        "D_comp": 600,
        "D_org": 784,
        "D_in": 600,
        "num_iters": 50,
        "L": 100,
        "lam": 2,
        "num_phis": 300,
        "num_epochs": 20,
        "batch_size": 5,
        "test_batch": 5,
        "lr": 1e-2,
        "shuffle": False,
        "classification": True,
        "info_period": 5,
        "device": device,
    }
