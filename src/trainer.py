"""
Copyright (c) 2020 Bahareh Tolooshams

code related to MNIST Classification, etc.

:author: Thomas Change & Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

import sys

sys.path.append("src/")

import utils


def train_ae_simulated(net, data_loader, hyp, criterion, optimizer, real_H, PATH):

    num_epochs = hyp["num_epochs"]
    device = hyp["device"]
    info_period = hyp["info_period"]

    err = []
    for epoch in range(num_epochs):
        for idx, code in tqdm(enumerate(data_loader)):

            img = torch.matmul(real_H, code.reshape(-1, net.D_enc, 1))
            img = img.to(device)
            # ===================forward=====================
            img_hat, _ = net(img)
            loss = criterion(img_hat, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.normalize()

            if idx % info_period == 0:
                print("loss:{:.4f}".format(loss.item()))

        err.append(utils.err_H(real_H, net.H.data))
        # ===================log========================

        print(
            "epoch [{}/{}], loss:{:.4f}, err_H:{:.4f}".format(
                epoch + 1, num_epochs, loss.item(), err[-1]
            )
        )

        torch.save(err[-1], os.path.join(PATH, "err_epoch{}.pt".format(epoch)))

    return err


def train_randproj_ae_simulated(
    net, data_loader, hyp, criterion, optimizer, real_H, phi, PATH, test_loader=None
):

    num_epochs = hyp["num_epochs"]
    device = hyp["device"]
    info_period = hyp["info_period"]

    err = []
    min_errH = 1
    bestH = None
    last_test_loss = 0
    true_decoder = torch.matmul(phi, real_H)
    true_decoder.requires_grad = False

    # guarantee net() takes a tuple in forward pass
    if len(phi.size()) == 2:
        true_decoder = true_decoder.unsqueeze(0)

    for epoch in range(num_epochs):
        for i, sample in tqdm(enumerate(data_loader)):
            # use ith phi to encode and decode
            i = i % true_decoder.size(0)

            sample, true_decoder = sample.to(device), true_decoder.to(device)

            img = torch.matmul(true_decoder[i], sample.view(-1, net.D_enc, 1)).view(
                -1, net.D_in, 1
            )
            # ===================forward=====================
            if len(phi.size()) == 2:
                img_hat, _ = net(img)
            else:
                img_hat, _ = net((i, img))
            loss = criterion(img_hat, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.normalize()

            if idx % info_period == 0:
                print("loss:{:.4f}".format(loss.item()))

        err.append(utils.err_H(real_H.cpu(), net.H.cpu().data))

        if test_loader != None:
            for i, sample in tqdm(enumerate(test_loader)):
                # use ith phi to encode and decode
                i = i % true_decoder.size(0)

                sample, true_decoder = sample.to(device), true_decoder.to(device)

                img = torch.matmul(true_decoder[i], sample.view(-1, net.D_enc, 1)).view(
                    -1, net.D_in, 1
                )
                # ===================forward=====================
                if len(phi.size()) == 2:
                    img_hat, _ = net(img)
                else:
                    img_hat, _ = net((i, img))
                test_loss = criterion(img_hat, img)

        # ===================log========================
        if err[-1] < min_errH:
            min_errH = err[-1]
            net.bestH = net.H.data
        if test_loader == None:
            print(
                "epoch [{}/{}], loss:{:.4f}, err_H:{:.4f}".format(
                    epoch + 1, num_epochs, loss.data, err[-1]
                )
            )
        else:
            print(
                "epoch [{}/{}], loss:{:.4f}, test_loss:{:.4f}, err_H:{:.4f}".format(
                    epoch + 1, num_epochs, loss.data, test_loss.data, err[-1]
                )
            )

        if test_loader != None:
            if np.abs(test_loss.data - last_test_loss) < 5e-4:
                return err
            else:
                last_test_loss = test_loss.data

        torch.save(err[-1], os.path.join(PATH, "err_epoch{}.pt".format(epoch)))
        torch.save(loss.item(), os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

    return err


def train_ae(net, data_loader, hyp, criterion, optimizer, PATH):

    num_epochs = hyp["num_epochs"]
    device = hyp["device"]
    info_period = hyp["info_period"]

    err = []
    min_err = None
    for epoch in range(num_epochs):
        for idx, (img, c) in tqdm(enumerate(data_loader)):

            img = img.to(device)
            data = img.view(-1, net.D_org, 1)

            if len(net.phi.size()) == 3:
                i = idx % net.phi.size(0)

            # ===================forward=====================
            output = net((i, data))
            loss = criterion(output[0], torch.matmul(net.phi[i], data))
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.normalize()

            if idx % info_period == 0:
                print("loss:{:.4f}".format(loss.item()))

        # ===================log========================

        if min_err is None or min_err >= loss.data:
            min_err = loss.item()
            net.bestH = net.H.cpu().data
        err.append(loss.item())
        print("epoch [{}/{}], loss:{:.4f} ".format(epoch + 1, num_epochs, loss.item()))

        torch.save(loss.item(), os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

    return err


def train_classifier_encodings(
    net, data_loader, hyp, criterion, optimizer, val_loader=None, getHs=False
):

    num_epochs = hyp["num_epochs"]
    device = hyp["device"]
    info_period = hyp["info_period"]

    train_acc = []
    val_acc = []
    Hs = []
    for epoch in tqdm(range(num_epochs)):
        for idx, (img, c) in enumerate(data_loader):
            img = img.to(device)
            c = c.to(device)

            if len(net.phi.size()) == 3:
                i = idx % net.phi.size(0)
            # ===================forward=====================

            output = net((i, img))

            loss = criterion(output, c)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.normalize()

            if idx % info_period == 0:
                print("loss:{:.4f}".format(loss.item()))

        # ===================log========================
        train_acc.append(test_network(data_loader, net, hyp))
        val_acc.append(test_network(val_loader, net, hyp))
        Hs.append(net.H.cpu().data)
        print(
            "epoch [{}/{}], loss:{:.4f}, train acc:{:.4f}, val acc:{:.4f}".format(
                epoch + 1, num_epochs, loss.item(), train_acc[-1], val_acc[-1]
            )
        )
    if getHs:
        return train_acc, val_acc, Hs
    return train_acc, val_acc


def test_network(data_loader, net, hyp, getExamples=False, getClasses=False):

    device = hyp["device"]

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        correct_ex = []
        incorrect_ex = []
        examples = 300
        for idx, (img, c) in tqdm(enumerate(data_loader)):

            img = img.to(device)
            c = c.to(device)

            img = img.view(-1, net.D_enc, 1)

            i = idx % net.phi.size(0)
            # ===================forward=====================
            output = net((i, img))

            correct_indicators = output.max(1)[1].data == c
            num_correct += correct_indicators.sum().item()
            num_total += c.size()[0]

            if getExamples:
                count = 0
                for j, indicator in enumerate(correct_indicators):
                    if indicator and len(correct_ex) <= examples:
                        correct_ex.append((i, img[j], c[j]))
                    elif not indicator and len(incorrect_ex) <= examples:
                        incorrect_ex.append((i, img[j], c[j]))
                    count += 1
                    if count > 4:
                        break
            if getClasses:
                correct
        # ===================log========================

    acc = num_correct / num_total
    if getExamples:
        return (acc, correct_ex, incorrect_ex)
    return acc
