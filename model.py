import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):

    def __init__(self, h, w, actions_num, in_channels=2):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 32, kernel_size=3, stride=2)
        self.linear_input_size = 32 * 7
        self.fc1 = nn.Linear(self.linear_input_size, 1024)
        self.fc2 = nn.Linear(1024, actions_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

