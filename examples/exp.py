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

