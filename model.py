import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    # def __init__(self, actions_num, in_channels=2):
    #     super(DQN, self).__init__()
    #     self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=1)
    #     # self.bn1 = nn.BatchNorm2d(32)
    #     self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
    #     # self.bn2 = nn.BatchNorm2d(32)
    #     self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
    #     # self.bn3 = nn.BatchNorm2d(32)
    #     self.linear_input_size = 64 * 10
    #     self.fc1 = nn.Linear(self.linear_input_size, 512)
    #     self.fc2 = nn.Linear(512, actions_num)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = x.view(-1, self.linear_input_size)
    #     x = F.relu(self.fc1(x))
    #     return self.fc2(x)

    def __init__(self, state_inputs, actions_num, in_channels=2):
        super(DQN, self).__init__()
        self.linear_input_size = state_inputs * in_channels
        self.fc_layers = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(256, actions_num)
        )

    def forward(self, x):
        x = x.view(-1, self.linear_input_size)
        x = self.fc_layers(x)
        return x


class Dueling_DQN(nn.Module):
    def __init__(self, state_inputs, actions_num, in_channels=2):
        super(Dueling_DQN, self).__init__()
        self.linear_input_size = state_inputs * in_channels
        self.fc_layers = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.05),
        )
        self.state_fc = nn.Linear(256, 1)
        self.advantage_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(256, actions_num)
        )

    def forward(self, x):
        x = x.view(-1, self.linear_input_size)
        x = self.fc_layers(x)
        v = self.state_fc(x)
        advantage = self.advantage_fc(x)
        advantage -= advantage.mean()
        return v + advantage


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, action_dim)
        )
        self.max_action = max_action
    

    def forward(self, x):
        x = self.fc_layers(x)
        action = torch.tanh(x) * self.max_action
        return action


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc_s = nn.Linear(state_dim, 512)
        self.fc_a = nn.Linear(action_dim, 512)
        self.fc_q = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(256, 1),
        )
        self.fc3 = nn.Linear(512, 1)
    
    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q =  self.fc_q(cat)
        return q



class EasyLinear(nn.Module):
    def __init__(self, dim, use_cuda):
        self.dim = dim
        self.use_cuda = use_cuda
        super(EasyLinear, self).__init__()
        self.fc_layers = nn.Sequential(
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
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()
        return self.fc_layers(x)
    

class VAE(nn.Module):
    def __init__(self, dim, latent_size, condition=False, num_labels=0):
        super(VAE, self).__init__()
        if condition == False:
            num_labels = 0
        self.condition = condition
        self.num_labels = num_labels
        self.dim = dim
        # Define encoder
        self.features = nn.Sequential(
          nn.Linear(dim + num_labels, 512),
          nn.ReLU(),
          nn.Linear(512, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU()
        )
        self.fc2mu = nn.Linear(512, latent_size)
        self.fc2Logvar = nn.Linear(512, latent_size)

        self.features_to_img = nn.Sequential(
            nn.Linear(latent_size+ num_labels, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
    
    def idx2oneHot(self, idx):
        assert torch.max(idx).item() < self.num_labels
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        
        onehot = torch.zeros(idx.size(0), self.num_labels)
        onehot.scatter_(1, idx.cpu().long(), 1)

        return onehot.cuda()    

    def encode(self, x, c):
        if self.condition:
            c = self.idx2oneHot(c)
            x = torch.cat((x, c), dim=-1)
        x = self.features(x)
        return self.fc2mu(x), self.fc2Logvar(x)
    
    def reparmeterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        if self.condition:
            c = self.idx2oneHot(c)
            z = torch.cat((z, c), dim=-1)
        res = self.features_to_img(z)
        return res

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparmeterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class ConvVAE(nn.Module):
    def __init__(self, latent_size, input_channel=4, condition=False, num_labels=2):
        """
        Shape: 816 + 531 = (1347, 4) + num_labels ==> 1349, 4 + one zeros ==> 1350, 4
        """
        super(ConvVAE, self).__init__()
        if condition == False:
            num_labels = 0
        self.condition = condition
        self.num_labels = num_labels
        self.input_channel = input_channel
        self.hidden_dim = 512
        self.reside_size = 25
        # Define encoder
        self.features = nn.Sequential(
            nn.Conv1d(input_channel, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(2),

            nn.Conv1d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(3),
        )
        self.features_to_hidden = nn.Sequential(
            nn.Linear(128 * self.reside_size, self.hidden_dim),
            nn.ReLU()
        )
        self.fc2mu = nn.Linear(self.hidden_dim, latent_size)
        self.fc2Logvar = nn.Linear(self.hidden_dim, latent_size)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_size + num_labels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 128 * self.reside_size),
            nn.ReLU()
        ) 
        self.features_to_img = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 3, 3),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, 64, 3, 3),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.ConvTranspose1d(32, 32, 3, 3),
            nn.Conv1d(32, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(10),

            nn.ConvTranspose1d(10, 10, 2, 2),
            nn.Conv1d(10, input_channel, 3),
            nn.ReLU(),
            nn.BatchNorm1d(input_channel),
        )
    
    def idx2oneHot(self, idx, encode=True):
        assert torch.max(idx).item() < self.num_labels
        
        if encode:
            onehot = torch.zeros((idx.size(0), self.input_channel, self.num_labels))
            for i, v in enumerate(idx.cpu().long()):
                onehot[i, :, v] = torch.ones(self.input_channel)
        else:
            if idx.dim() == 1:
                idx = idx.unsqueeze(1)
        
            onehot = torch.zeros(idx.size(0), self.num_labels)
            onehot.scatter_(1, idx.cpu().long(), 1)
        return onehot.cuda()    

    def encode(self, x, c):
        if self.condition:
            c = self.idx2oneHot(c)
            x = torch.cat((x, c), dim=-1)
        x = self.features(x)
        x = x.view(-1, 128 * self.reside_size)
        x = self.features_to_hidden(x)
        return self.fc2mu(x), self.fc2Logvar(x)
    
    def reparmeterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        if self.condition:
            c = self.idx2oneHot(c, encode=False)
            z = torch.cat((z, c), dim=-1)
        features = self.latent_to_features(z).view(-1, 128, self.reside_size)
        return self.features_to_img(features)

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparmeterize(mu, logvar)
        return self.decode(z, c), mu, logvar