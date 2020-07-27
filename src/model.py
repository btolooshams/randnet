"""
Copyright (c) 2020 Bahareh Tolooshams

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils


class CRsAEDense(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAEDense, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.lam = hyp["lam"]
        self.D_in = hyp["D_in"]
        self.D_enc = hyp["D_enc"]
        self.device = hyp["device"]

        if H is None:
            self.H = torch.nn.Parameter(
                F.normalize(torch.randn(self.D_in, self.D_enc), dim=0)
            )
        else:
            self.H = torch.nn.Parameter(H)

        self.H = self.H.to(self.device)

        self.relu = torch.nn.ReLU()

    def normalize(self):
        self.H.data = F.normalize(self.H.data, dim=0)

    def forward(self, x):
        num_batches = x.shape[0]

        x_old = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        yk = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        x_new = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()
        for t in range(self.T):
            H_wt = x - torch.matmul(self.H, yk.reshape(-1, self.D_enc, 1))
            x_new = yk + torch.matmul(torch.t(self.H), H_wt) / self.L
            x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = torch.matmul(self.H, x_new)

        return z, x_new


class CRsAERandProj(torch.nn.Module):
    def __init__(self, hyp, H=None, phi=None):
        super(CRsAERandProj, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.lam = hyp["lam"]
        self.D_in = hyp["D_in"]
        self.D_org = hyp["D_org"]
        self.D_enc = hyp["D_enc"]
        self.device = hyp["device"]
        self.eval_mode = False

        if H is None:
            self.H = F.normalize(torch.randn(self.D_org, self.D_enc), dim=0)
        else:
            self.H = H

        if phi is None:
            self.phi = F.normalize(torch.randn(1, self.D_in, self.D_org), dim=0)
        else:
            self.phi = phi

        self.H = torch.nn.Parameter(self.H)
        self.phi = torch.nn.Parameter(self.phi)
        self.phi.requires_grad = False

        self.H = self.H.to(self.device)
        self.phi = self.phi.to(self.device)

        self.relu = torch.nn.ReLU()

    def normalize(self):
        self.H.data = F.normalize(self.H.data, dim=0)

    def forward(self, x):

        # if testing use the H with the lowest err_H
        if self.eval_mode:
            H = self.bestH
        else:
            H = self.H

        # for multiple phi use ith phi for image x
        if isinstance(x, tuple):
            i, x = x
            phiH = torch.matmul(self.phi[i], H)
        else:
            phiH = torch.matmul(self.phi, H)

        num_batches = x.shape[0]

        x_old = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        yk = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        x_new = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()

        phiH = phiH.to(self.device)

        for t in range(self.T):
            H_wt = x - torch.matmul(phiH, yk.reshape(-1, self.D_enc, 1))
            x_new = yk + torch.matmul(torch.t(phiH), H_wt) / self.L
            x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = torch.matmul(phiH, x_new)

        return z, x_new


class CRsAERandProjClassifier(torch.nn.Module):
    def __init__(self, hyp, H=None, phi=None):
        super(CRsAERandProjClassifier, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.lam = hyp["lam"]
        self.D_in = hyp["D_in"]
        self.D_org = hyp["D_org"]
        self.D_enc = hyp["D_enc"]
        self.device = hyp["device"]
        self.eval_mode = False

        if H is None:
            self.H = F.normalize(torch.randn(self.D_org, self.D_enc), dim=0)
        else:
            self.H = H

        if phi is None:
            self.phi = F.normalize(torch.randn(1, self.D_in, self.D_org), dim=0)
        else:
            self.phi = phi

        self.H = torch.nn.Parameter(self.H)
        self.phi = torch.nn.Parameter(self.phi)
        self.phi.requires_grad = False

        self.H = self.H.to(self.device)
        self.phi = self.phi.to(self.device)

        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(self.D_enc, 10)

        # set the classifier on the gpu
        # to be done

    def normalize(self):
        self.H.data = F.normalize(self.H.data, dim=0)

    def forward(self, x):

        # if testing use the H with the lowest err_H
        if self.eval_mode:
            H = self.bestH
        else:
            H = self.H

        # for multiple phi use ith phi for image x
        if isinstance(x, tuple):
            i, x = x
            phiH = torch.matmul(self.phi[i], H)
            x = torch.matmul(self.phi[i], x)
        else:
            phiH = torch.matmul(self.phi, H)
            x = torch.matmul(self.phi, x)

        num_batches = x.shape[0]

        x_old = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        yk = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        x_new = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()

        phiH = phiH.to(self.device)

        for t in range(self.T):

            H_wt = x - torch.matmul(phiH, yk.view(-1, self.D_enc, 1))
            x_new = yk + torch.matmul(torch.t(phiH), H_wt) / self.L
            x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        return self.classifier(x_new.view(-1, self.D_enc))


class CRsAERandProjAeClassifier(torch.nn.Module):
    def __init__(self, hyp, H=None, phi=None):
        super(CRsAERandProjAeClassifier, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.lam = hyp["lam"]
        self.D_in = hyp["D_in"]
        self.D_org = hyp["D_org"]
        self.D_enc = hyp["D_enc"]
        self.device = hyp["device"]
        self.eval_mode = False

        if H is None:
            self.H = F.normalize(torch.randn(self.D_org, self.D_enc), dim=0)
        else:
            self.H = H

        if phi is None:
            self.phi = F.normalize(torch.randn(1, self.D_in, self.D_org), dim=0)
        else:
            self.phi = phi

        self.H = torch.nn.Parameter(self.H)
        self.phi = torch.nn.Parameter(self.phi)
        self.phi.requires_grad = False

        self.H = self.H.to(self.device)
        self.phi = self.phi.to(self.device)

        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(self.D_enc, 10)
        self.encoding_mode = False

    def normalize(self):
        self.H.data = F.normalize(self.H.data, dim=0)

    def forward(self, x):
        if self.encoding_mode:
            i, x = x
            return self.classifier(x.view(-1, self.D_enc))

        # if testing use the H with the lowest err_H
        if self.eval_mode:
            H = self.bestH
        else:
            H = self.H

        # for multiple phi use ith phi for image x
        if isinstance(x, tuple):
            i, x = x
            phiH = torch.matmul(self.phi[i], H)
            x = torch.matmul(self.phi[i], x)
        else:
            phiH = torch.matmul(self.phi, H)
            x = torch.matmul(self.phi, x)

        num_batches = x.shape[0]

        x_old = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        yk = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        x_new = torch.zeros(num_batches, self.D_enc, 1, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()

        phiH = phiH.to(self.device)

        for t in range(self.T):

            H_wt = x - torch.matmul(phiH, yk.view(-1, self.D_enc, 1))
            x_new = yk + torch.matmul(torch.t(phiH), H_wt) / self.L
            x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = torch.matmul(phiH, x_new)
        return (z, x_new, self.classifier(x_new.view(-1, self.D_enc)))
