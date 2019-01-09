#!/usr/bin/env python3
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data as Data

import hyperparam as hparams

if hparams.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if hparams.plot:
    import matplotlib.pyplot as plt

    os.makedirs('output/%s' % hparams.method, exist_ok=True)
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    plt.show(block=False)


def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class ODEFunc(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ODEFunc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, t, x):
        x = self.fc(x)
        return x


def get_data(start=-5, end=5, steps=100):
    X = torch.empty(steps * steps, 2)
    x = torch.linspace(start, end, steps)
    y = torch.linspace(start, end, steps)
    for i in range(steps):
        for j in range(steps):
            X[i * steps + j, 0] = x[i]
            X[i * steps + j, 1] = y[j]
    Y = torch.empty(steps * steps)
    for i in range(steps * steps):
        Y[i] = int(torch.sum(X[i] ** 2) <= 9)
    return X, Y.long()


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_time, rtol=hparams.tolerance, atol=hparams.tolerance,
                     method=hparams.method)
        return out[-1]


def plot(X, y, it):
    plt.scatter(X.data.numpy()[:, 0], X.data.numpy()[:, 1], c=y, s=30, cmap='RdYlGn')
    plt.savefig(os.path.join('output/%s' % hparams.method, '%d.png' % it))
    plt.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    lr = hparams.lr
    model = ODEBlock(odefunc=ODEFunc(2, 10, 10, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    X, y = get_data()
    sampler = Data.BatchSampler(Data.RandomSampler(Data.TensorDataset(X, y)), batch_size=2000,
                                drop_last=False)
    iterator = iter(sampler)
    indices = torch.tensor(next(iterator))
    X_train, y_train = torch.index_select(X, 0, indices), torch.index_select(y, 0, indices)
    for it in range(hparams.niters + 1):
        model.train()
        optimizer.zero_grad()
        result = model(X_train)
        loss = loss_func(result, y_train)
        loss.backward()
        if it % hparams.print_every == 0:
            print('iter %d: %f' % (it, loss))
        optimizer.step()

        if it % hparams.test_every == 0:
            model.eval()
            result = model(X)
            prediction = torch.max(F.softmax(result, dim=1), 1)[1].long()
            accuracy = torch.mean((prediction == y).float())
            print('Acc: %f' % accuracy)
            if hparams.plot:
                plot(X, prediction, it)

        if it % hparams.decay_every == 0:
            lr = lr * 0.99
            adjust_learning_rate(optimizer, lr)
            print('current lr: %f' % lr)
