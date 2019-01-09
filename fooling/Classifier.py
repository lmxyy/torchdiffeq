import torch
import torch.nn as nn


# import matplotlib.pyplot as plt

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Linear(4 * 4 * 64, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        # x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
