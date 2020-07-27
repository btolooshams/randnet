"""
Copyright (c) 2020 Bahareh Tolooshams

utils

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np


def normalize(x):
    x_normed = x / x.norm(dim=0, keepdim=True)
    return x_normed


def err_H(H, H_hat):
    err = 0
    for i in range(H.size()[1]):
        err_i = 1 - np.dot(H[:, i], H_hat[:, i]) ** 2
        if err_i > err:
            err = err_i
    return err


def err_H_min(H, H_hat):
    err = 1
    for i in range(H.size()[1]):
        err_i = 1 - np.dot(H[:, i], H_hat[:, i]) ** 2
        if err_i < err:
            err = err_i
    return err


def err_H_avg(H, H_hat):
    err = 0
    for i in range(H.size()[1]):
        err_i = 1 - np.dot(H[:, i], H_hat[:, i]) ** 2
        err += err_i
    return err / H.size()[1]


def err_H_all(H, H_hat):
    errs = []
    for i in range(H.size()[1]):
        err_i = 1 - np.dot(H[:, i], H_hat[:, i]) ** 2
        errs.append(err_i)
    return errs


def sample_var(dataset, real_H):
    return np.dot(dataset.samples.T, real_H.t()).var(1).mean()


def display_imgs(net, test_sparse, real_H, D_in):
    img = torch.matmul(test_sparse.view(1, -1), real_H.t()).view(-1, D_in, 1)
    plt.plot(img.flatten().data.numpy())
    comp_img = torch.matmul(net.phi.cpu().data, img)
    net(comp_img)
    plt.plot(torch.matmul(net.H.cpu(), net.last_encoding[0]).flatten().data.numpy())
    plt.legend(["Real image", "Recovered image"])
    plt.show()


def display_img_enc(net, real_H, dataset):
    i = 0
    net.eval_mode = True
    net.use_cuda = False
    recon_img = net(
        torch.matmul(torch.matmul(net.phi.cpu().data, real_H.cpu().data), dataset[i][0])
    ).view(1, -1)
    display_imgs(net, dataset[i][0], real_H.cpu(), net.D_org)
    plt.scatter(range(net.D_enc), dataset[i][0])
    plt.scatter(range(net.D_enc), net.last_encoding[0].cpu())
    plt.title("Learned H encodings lam = " + str(net.lam))
    plt.legend(["Real encoding", "Recovered encoding"])
    plt.show()

    net.use_cuda = True


def display_err_plot(errs, initial_err):
    plt.plot(range(len(errs) + 1), [initial_err] + list(errs))
    plt.title("Err vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Err")
    plt.show()


def display_plots(net, real_H, dataset, errs, initial_err):
    display_img_enc(net, real_H, dataset)
    display_err_plot(errs, initial_err)


def save_model(net, acc, initial_H, name, num_iters, lam, mse):
    torch.save(
        net.H.data,
        name
        + "_Din"
        + str(net.D_in)
        + "_Denc"
        + str(net.D_enc)
        + "_iters"
        + str(num_iters)
        + "_lam"
        + str(lam)
        + "H.pt",
    )
    torch.save(
        net.classifier,
        name
        + "_Din"
        + str(net.D_in)
        + "_Denc"
        + str(net.D_enc)
        + "_iters"
        + str(num_iters)
        + "_lam"
        + str(lam)
        + "classifier.pt",
    )
    torch.save(
        torch.tensor(acc[0]),
        name
        + "_Din"
        + str(net.D_in)
        + "_Denc"
        + str(net.D_enc)
        + "_iters"
        + str(num_iters)
        + "_lam"
        + str(lam)
        + "TrainAcc.pt",
    )
    torch.save(
        torch.tensor(acc[1]),
        name
        + "_Din"
        + str(net.D_in)
        + "_Denc"
        + str(net.D_enc)
        + "_iters"
        + str(num_iters)
        + "_lam"
        + str(lam)
        + "TestAcc.pt",
    )
    torch.save(
        torch.tensor(mse),
        name
        + "_Din"
        + str(net.D_in)
        + "_Denc"
        + str(net.D_enc)
        + "_iters"
        + str(num_iters)
        + "_lam"
        + str(lam)
        + "MSE.pt",
    )
    torch.save(
        initial_H,
        name
        + "_Din"
        + str(net.D_in)
        + "_Denc"
        + str(net.D_enc)
        + "_iters"
        + str(num_iters)
        + "_lam"
        + str(lam)
        + "initial_H.pt",
    )
