<<<<<<< HEAD
#!/usr/bin/env python
import torch
from torch import nn

from torchdiffeq import odeint


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y ** 3)


class ODEBlackbox:
    def __init__(self, method):
        self.method = method

    def calc(self,y):
        return odeint(func, y, t=torch.linspace(0., 25., 10), method=self.method)


func = ODEFunc()
model1 = ODEBlackbox('dopri5')
model2 = ODEBlackbox('mh')
model3 = ODEBlackbox('euler')

if __name__ == '__main__':
    y = torch.rand(1,2)
    print(model1.calc(y))
    print(model2.calc(y))
    print(model3.calc(y))

=======
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--tol', type=float, default=1e-3)
args = parser.parse_args()

METHOD = 'dopri5'


def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


def get_data():
    with torch.no_grad():
        n_data = torch.zeros(100, 2)
        x_normal = torch.normal(n_data, 1)

        norm = torch.sqrt(torch.sum((x_normal * x_normal), 1, True))
        x0 = x_normal / 10 + x_normal / norm + 5
        x1 = x_normal / 10 + x_normal / (norm / 1.5) + 5
        y0 = torch.zeros(100)
        y1 = torch.ones(100)

        x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

        y = torch.cat((y0, y1)).type(torch.LongTensor)
    return x, y


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 2),
        )

    def forward(self, t, x):
        x = self.fc(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method=METHOD)
        return out[1]


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    USE_ADJ = False
    if USE_ADJ:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    model = ODEBlock(odefunc=ODEFunc().apply(initialize_weights)).to(device)

    LR = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(device)

    for i in range(5000):
        model.train()
        x, y = get_data()
        target_y = y.data.numpy()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        result = model(x)
        loss = loss_func(result, y)
        loss.backward()
        optimizer.step()

        if (i % 100 == 0):
            LR = LR * 0.999
            adjust_learning_rate(optimizer, LR)

        if (i % 200 == 0):
            print('iter' + str(i))
            prediction = torch.max(F.softmax(result, dim=1), 1)[1]
            pred_y = prediction.cpu().data.numpy().squeeze()
            accuracy = sum(pred_y == target_y) / 200

            print('Accuracy=%.2f' % accuracy)

    model.eval()
    num_row = 50
    num_column = 50

    x0 = torch.linspace(-2.5, 2.5, num_row) + 5
    x1 = torch.linspace(-2.5, 2.5, num_column) + 5

    x = torch.zeros(num_row * num_column, 2)

    for i in range(num_row):
        for j in range(num_column):
            x[i * num_row + j, 0] = x0[i]
            x[i * num_row + j, 1] = x1[j]

    with torch.no_grad():
        x = x.to(device)
        result = model(x)
        prediction = torch.max(F.softmax(result, dim=1), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        plt.scatter(x.cpu().data.numpy()[:, 0], x.cpu().data.numpy()[:, 1], c=pred_y, s=30, cmap='RdYlGn')
        plt.savefig(METHOD + '.png')
>>>>>>> xzhbranch
