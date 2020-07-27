"""
Copyright (c) 2020 Bahareh Tolooshams

train

:author: Thomas Chang & Bahareh Tolooshams
"""


import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import os
from datetime import datetime
from sacred import Experiment

import sys

sys.path.append("src/")

import model, generator, trainer, utils, conf

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("train", ingredients=[config_ingredient])


@ex.automain
def run(cfg):

    hyp = cfg["hyp"]

    print(hyp)

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    os.makedirs(PATH)

    filename = os.path.join(PATH, "hyp.pickle")
    with open(filename, "wb") as file:
        pickle.dump(hyp, file)

    print("load data.")
    if hyp["dataset"] == "MNIST":
        train_loader, test_loader = generator.get_MNIST_loaders(
            hyp["batch_size"], shuffle=hyp["shuffle"]
        )
        phis = F.normalize(
            torch.randn(hyp["num_phis"], hyp["D_comp"], hyp["D_org"]), dim=1
        )
        H_init = None
    elif hyp["dataset"] == "simulated":
        real_H, H_init, phis, train_loader = generator.generate_simulated_data(hyp)
    else:
        print("ERROR: dataset loader is not implemented.")

    print("create model.")
    if hyp["network"] == "CRsAEDense":
        net = model.CRsAEDense(hyp, H_init)
    elif hyp["network"] == "CRsAERandProj":
        net = model.CRsAERandProj(hyp, H_init, phis)
    elif hyp["network"] == "CRsAERandProjClassifier":
        net = model.CRsAERandProjClassifier(hyp, H_init, phis)
    elif hyp["network"] == "CRsAERandProjAeClassifier":
        net = model.CRsAERandProjAeClassifier(hyp, H_init, phis)
    else:
        print("model does not exist!")

    torch.save(net.H, os.path.join(PATH, "H_init.pt"))

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyp["lr"], eps=1e-3)

    if hyp["classification"]:
        net.H.requires_grad = True
        net.classifier.requires_grad = False

    print("train auto-encoder.")
    if hyp["dataset"] == "simulated":
        if hyp["network"] == "CRsAEDense":
            err = trainer.train_ae_simulated(
                net, train_loader, hyp, criterion, optimizer, real_H, PATH
            )
        elif hyp["network"] == "CRsAERandProj":
            err = trainer.train_randproj_ae_simulated(
                net, train_loader, hyp, criterion, optimizer, real_H, phis, PATH
            )

    else:
        err = trainer.train_ae(net, train_loader, hyp, criterion, optimizer, PATH)

    if hyp["classification"]:
        net.H.requires_grad = False
        net.classifier.requires_grad = True

        optimizer.zero_grad()
        enc_tr_loader, enc_te_loader = generator.get_encoding_loaders(
            train_loader, test_loader, net, hyp
        )

        criterion_class = torch.nn.CrossEntropyLoss()

        print("train classifier.")
        net.encoding_mode = True
        acc = trainer.train_classifier_encodings(
            net, enc_tr_loader, hyp, criterion_class, optimizer, enc_te_loader
        )

        final_acc = (
            trainer.test_network(train_loader, net, hyp),
            trainer.test_network(test_loader, net, hyp),
        )
        net.encoding_mode = False
        print("final_acc", final_acc)
