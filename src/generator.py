"""
Copyright (c) 2020 Bahareh Tolooshams

data generator

:author: Thomas Chang & Bahareh Tolooshams
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("src/")

import utils


def get_MNIST_loaders(batch_size, shuffle=False, train_batch=None, test_batch=None):
    if train_batch == None:
        train_loader = get_MNIST_loader(batch_size, trainable=True, shuffle=shuffle)
    else:
        train_loader = get_MNIST_loader(train_batch, trainable=True, shuffle=shuffle)

    if test_batch == None:
        test_loader = get_MNIST_loader(batch_size, trainable=False, shuffle=shuffle)
    else:
        test_loader = get_MNIST_loader(test_batch, trainable=False, shuffle=shuffle)
    return train_loader, test_loader


def get_MNIST_loader(batch_size, trainable=True, shuffle=False):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "../data",
            train=trainable,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


class SparseVectorDataset(Dataset):
    def __init__(self, n, dim, ones, transform=None, seed=None):
        self.samples = generate_sparse_samples(n, dim, ones, seed)
        self.transform = transform

    def __len__(self):
        return self.samples.shape[1]

    def __getitem__(self, idx):
        sample = self.samples[:, idx].reshape(-1, 1, 1)
        if self.transform:
            sample = self.transform(sample).float()

        return sample


class SparseCompImageDataset(Dataset):
    def __init__(self, n, dim, ones, real_H, phi, transform=None):
        self.sparse_vectors = generate_sparse_samples(n, dim, ones)
        print(self.sparse_vectors.shape)
        self.comp_img = np.dot(real_H, self.sparse_vectors)
        self.img = np.dot(phi, self.comp_img)
        self.samples = np.dot(phi.T, self.img)
        self.transform = transform

    def __len__(self):
        return self.samples.shape[1]

    def __getitem__(self, idx):
        sample = self.samples[:, idx].reshape(-1, 1, 1)
        img = self.img[:, idx].reshape(-1, 1, 1)
        if self.transform:
            sample = self.transform(sample).float()
            img = self.transform(img).float()
        return sample, img


class EncodingDataset(Dataset):
    def __init__(self, data_loader, net, device=None, transform=None, seed=None):
        self.samples = []
        self.c = []
        print("create encoding dataset.")
        for idx, (img, c) in tqdm(enumerate(data_loader)):
            img = img.to(device)
            img = img.view(-1, net.D_org, 1)

            if len(net.phi.size()) == 3:
                i = idx % net.phi.size(0)

            _, enc, _ = net((i, img))

            self.samples.append(enc)
            self.c.append(c)

        self.samples = torch.cat(self.samples)
        self.c = torch.cat(self.c)
        self.D_enc = net.D_enc
        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx].reshape(-1, self.D_enc, 1)

        if self.transform:
            sample = self.transform(sample).float()

        return sample, self.c[idx]


def generate_sparse_samples(n, dim, ones, seed=None, unif=True):
    samples = np.zeros((n, dim))
    np.random.seed(seed)
    for i in range(n):
        ind = np.random.choice(dim, ones, replace=False)
        if unif:
            # draws amplitude from [-5,-4] U [4,5] uniformly
            samples[i][ind] = np.random.uniform(4, 5, ones) * (
                (np.random.uniform(0, 1, ones) > .5) * 2 - 1
            )
        else:
            # amplitude is 1 or -1 .5 prob of each
            samples[i][ind] = np.array([1] * ones) * (
                (np.random.uniform(0, 1, ones) > .5) * 2 - 1
            )
    return samples.T


def generate_sparse_phi(sparsity, num_phi, D_enc, D_img):
    phis = [
        torch.tensor(generate_sparse_samples(D_img, D_enc, sparsity, unif=False))
        .float()
        .t()
        for _ in range(num_phi)
    ]
    return torch.stack(phis)


def generate_simulated_data(hyp):
    seed = hyp["seed"]
    D_enc = hyp["D_enc"]
    D_org = hyp["D_org"]
    D_comp = hyp["D_comp"]
    sparsity = hyp["sparsity"]
    randomness = hyp["randomness"]
    num_phis = hyp["num_phis"]
    num_nonzero = hyp["num_nonzero"]
    num_samples = hyp["num_samples"]
    batch_size = hyp["batch_size"]

    torch.manual_seed(seed)
    real_H = utils.normalize(torch.randn(D_org, D_enc)).float()
    noise = utils.normalize(torch.randn(D_org, D_enc)) * randomness
    H_init = utils.normalize(real_H * (1 - randomness) + noise)
    phis = generate_sparse_phi(sparsity, num_phis, D_org, D_comp)

    dataset = SparseVectorDataset(
        num_samples,
        D_enc,
        num_nonzero,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        seed=seed,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return real_H, H_init, phis, data_loader


def get_encoding_loaders(train_loader, test_loader, net, hyp):
    train_dataset = EncodingDataset(train_loader, net, hyp["device"])
    test_dataset = EncodingDataset(test_loader, net, hyp["device"])
    enc_tr_loader = DataLoader(train_dataset, batch_size=hyp["batch_size"])
    enc_te_loader = DataLoader(test_dataset, batch_size=hyp["batch_size"])
    return enc_tr_loader, enc_te_loader
