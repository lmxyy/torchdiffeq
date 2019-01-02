import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    def __init__(self, n_feature, n_hidden, n_output):
        super(ODEFunc, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        # print('second forward')
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x


if __name__ == '__main__':
    
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')    
    
    model = ODEFunc(2, 10, 2).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss().to(device)

    for i in range(300):
        model.train()
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
            print('Accuracy=%.2f' % accuracy)
            # plt.text(1.5, -4, 'Accuracy = %.2f' % accuracy, fontdicct = {'size':20, 'color':'red'})
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


