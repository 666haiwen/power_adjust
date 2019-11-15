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
    def __init__(self, dim, loads_dim, latent_size, condition=False, num_labels=0):
        super(VAE, self).__init__()
        if condition == False:
            num_labels = 0
        self.condition = condition
        self.num_labels = num_labels
        self.loads_dim = loads_dim
        self.dim = dim
        # Define encoder
        self.loads_features = nn.Sequential(
            nn.Linear(loads_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()  
        )
        self.solution_features = nn.Sequential(
          nn.Linear(dim - loads_dim + num_labels, 512),
          nn.ReLU(),
          nn.Linear(512, 1024),
          nn.ReLU(),
          nn.Linear(1024, 2048),
          nn.ReLU(),
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU()
        )
        self.fc2mu_loads = nn.Linear(512, latent_size[0])
        self.fc2Logvar_loads = nn.Linear(512, latent_size[0])
        self.fc2mu_solution = nn.Linear(512, latent_size[1])
        self.fc2Logvar_solution = nn.Linear(512, latent_size[1])

        self.features_to_img = nn.Sequential(
            nn.Linear(sum(latent_size) + num_labels, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
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

    def encode_loads(self, x):
        x = self.loads_features(x)
        return self.fc2mu_loads(x), self.fc2Logvar_loads(x)

    def encode_solution(self, x, c):
        if self.condition:
            x = torch.cat((x, c), dim=-1)
        x = self.solution_features(x)
        return self.fc2mu_solution(x), self.fc2Logvar_solution(x)
    
    def encode(self, x, c):
        loads = x[:, :self.loads_dim]
        solution = x[:, self.loads_dim:]
        mu_loads, logvar_loads = self.encode_loads(loads)
        mu_solution, logvar_solution = self.encode_solution(solution, c)
        mu = torch.cat((mu_loads, mu_solution), dim=-1)
        logvar = torch.cat((logvar_loads, logvar_solution), dim=-1)
        return mu, logvar

    def reparmeterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        if self.condition:
            z = torch.cat((z, c), dim=-1)
        res = self.features_to_img(z)
        return res

    def forward(self, x, c=None):
        if self.condition:
            c = self.idx2oneHot(c)
        mu, logvar = self.encode(x, c)
        z = self.reparmeterize(mu, logvar)
        return self.decode(z, c), mu, logvar
