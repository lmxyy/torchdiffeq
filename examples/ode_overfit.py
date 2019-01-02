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

def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_data():
    
    # 生成一系列的数据
    '''
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)
    y0 = torch.zeros(100)                   #类型0的标签
    x1 = torch.normal(-2 * n_data, 1)
    y1 = torch.ones(100)                    #类型1的标签

    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

    # x = x / 1000

    y = torch.cat((y0, y1)).type(torch.LongTensor)
    # print(y.size())
    # plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c = y.data.numpy(), s = 50, cmap = 'RdYlGn')
    # plt.show()
    return x, y
    '''
    with torch.no_grad():
        n_data = torch.zeros(100, 2)
        x_normal = torch.normal(n_data, 1)

        norm = torch.sqrt( torch.sum((x_normal * x_normal), 1, True) )
        x0 = x_normal / 10 + x_normal / norm
        x1 = x_normal / 10 + x_normal / (norm / 1.5)
        y0 = torch.zeros(100)                   #类型0的标签
        y1 = torch.ones(100)                    #类型1的标签

        x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

        # x = x / 1000

        y = torch.cat((y0, y1)).type(torch.LongTensor)
    return x, y

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            # nn.ReLU(),
        )
    
    def forward(self, t, x):
        # print('second forward')
        x = self.fc(x)
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
    USE_ADJ = False
    if USE_ADJ:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')    
    
    # model = ODEBlock(odefunc=ODEFunc(2, 10, 2)).to(device)
    model = ODEBlock(odefunc=ODEFunc()).to(device)
    LR = 1e-3
    optimizer = optim.RMSprop(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(device)

    for i in range(500):
        # LR = LR * 0.999
        # adjust_learning_rate(optimizer, LR)
        model.train()
        x, y = get_data()
        target_y = y.data.numpy()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        result = model(x)    
        loss = loss_func(result, y)
        loss.backward()
        optimizer.step()

        if (i % 50 == 0):
            #plt.cla()
            LR = LR * 0.99
            adjust_learning_rate(optimizer, LR)
            # try something interesting
            # with torch.no_grad():
               #   result = odeint(model.odefunc, x, torch.tensor([0, 5]).float(), rtol=args.tol, atol=args.tol)[1]
            
            prediction = torch.max(F.softmax(result, dim = 1), 1)[1]
            pred_y = prediction.cpu().data.numpy().squeeze()
            
            plt.scatter(x.cpu().data.numpy()[:, 0], x.cpu().data.numpy()[:, 1], c = pred_y, s = 30, cmap = 'RdYlGn')
            accuracy = sum(pred_y == target_y) / 200 # 统计预测的准确率
            
            print('Accuracy=%.2f' % accuracy)
            
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 15, 'color':  'red'})
            plt.show()
    # plt.ioff()
    # plt.show()

    model.eval()
    num_row = 50
    num_column = 50

    x0 = torch.linspace(-2.5, 2.5, num_row)
    x1 = torch.linspace(-2.5, 2.5, num_column)

    x = torch.zeros(num_row * num_column, 2)

    for i in range(num_row):
        for j in range(num_column):
            x[i * num_row + j, 0] = x0[i]
            x[i * num_row + j, 1] = x1[j]

    with torch.no_grad():
        x = x.to(device)
        result = model(x)
        prediction = torch.max(F.softmax(result, dim = 1), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        plt.scatter(x.cpu().data.numpy()[:, 0], x.cpu().data.numpy()[:, 1], c = pred_y, s = 30, cmap = 'RdYlGn')
        plt.show()


