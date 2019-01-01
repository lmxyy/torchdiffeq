import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def get_data():
# 生成一系列的数据
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)
    y0 = torch.zeros(100)                   #类型0的标签
    x1 = torch.normal(-2 * n_data, 1)
    y1 = torch.ones(100)                    #类型1的标签

    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    y = torch.cat((y0, y1)).type(torch.LongTensor)
    # print(y.size())
    # plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c = y.data.numpy(), s = 50, cmap = 'RdYlGn')
    # plt.show()
    return x, y

class ODEFunc(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(ODEFunc, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
    
    def forward(self, t, x):
        # print('second forward')
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        # print('first forward')
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
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

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')    
    
    model = ODEBlock(odefunc=ODEFunc(2, 10, 2)).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss().to(device)

    for i in range(300):
        x, y = get_data()
        target_y = y.data.numpy()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        result = model(x)    
        loss = loss_func(result, y)
        loss.backward()
        optimizer.step()

        if (i % 20 == 0):
            #plt.cla()
            prediction = torch.max(F.softmax(result, dim = 1), 1)[1]
            pred_y = prediction.cpu().data.numpy().squeeze()
            plt.scatter(x.cpu().data.numpy()[:, 0], x.cpu().data.numpy()[:, 1], c = pred_y, s = 30, cmap = 'RdYlGn')
            accuracy = sum(pred_y == target_y) / 200 # 统计预测的准确率
            # plt.text(1.5, -4, 'Accuracy = %.2f' % accuracy, fontdicct = {'size':20, 'color':'red'})
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 15, 'color':  'red'})
            plt.show()
    # plt.ioff()
    # plt.show()
