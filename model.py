import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# class DQN(nn.Module):

#     def __init__(self, actions_num, in_channels=2):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
#         self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
#         self.linear_input_size = 64 * 10
#         self.fc1 = nn.Linear(self.linear_input_size, 512)
#         self.fc2 = nn.Linear(512, actions_num)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, self.linear_input_size)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

class DQN(nn.Module):
    def __init__(self, state_inputs, actions_num, in_channels=2):
        super(DQN, self).__init__()
        self.linear_input_size = state_inputs * in_channels
        self.fc1 = nn.Linear(self.linear_input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, actions_num)

    def forward(self, x):
        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

