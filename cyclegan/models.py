import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, channel=2):
        super(Generator, self).__init__()

        # Initial convolution block       
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.InstanceNorm1d(channel),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.InstanceNorm1d(channel),
            nn.ReLU(inplace=True),

            nn.Linear(512, 2048),
            nn.InstanceNorm1d(channel),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 512),
            nn.InstanceNorm1d(channel),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.InstanceNorm1d(channel),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
