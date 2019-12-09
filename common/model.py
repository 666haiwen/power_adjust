import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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
    
class VAE(nn.Module):
    def __init__(self, latent_size, input_channel=2, condition=False, num_labels=2):
        super(VAE, self).__init__()
        if condition == False:
            num_labels = 0
        self.condition = condition
        self.num_labels = num_labels
        self.input_channel = input_channel
        self.input_dim = 86
        self.hidden_dim = 1024
        self.reside_size = 6
        self.reside_channel = 512

        # Define encoder
        self.features = nn.Sequential(
            # 48
            nn.Conv1d(input_channel + num_labels, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(48),

            # 24
            nn.Conv1d(64, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),

            # 12
            nn.Conv1d(128, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),

            # 6
            nn.Conv1d(256, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),
        )

        # latent
        self.features_to_hidden = nn.Sequential(
            nn.Linear(self.reside_channel * self.reside_size, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, self.hidden_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.fc2mu = nn.Linear(self.hidden_dim, latent_size)
        self.fc2Logvar = nn.Linear(self.hidden_dim, latent_size)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_size + num_labels, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, self.reside_channel * self.reside_size),
            nn.LeakyReLU(inplace=True),
        ) 

        # Decoder
        self.features_to_img = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 2, 2),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.ConvTranspose1d(256, 256, 2, 2),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 128, 2, 2),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, 64, 2, 2),
            nn.Conv1d(64, input_channel, 3, padding=1),
            nn.BatchNorm1d(input_channel),
        )
        self.init_weight()
    
    def init_weight(self):
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param.data)
            elif 'conv' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize liner transform
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize the batch norm layer
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
    
    def idx2oneHot(self, idx, encode=True):
        assert torch.max(idx).item() < self.num_labels
        
        if encode:
            onehot = torch.zeros((idx.size(0), self.num_labels, self.input_dim))
            for i, v in enumerate(idx.cpu().long()):
                onehot[i, v, :] = torch.ones(self.input_dim)
        else:
            if idx.dim() == 1:
                idx = idx.unsqueeze(1)
        
            onehot = torch.zeros(idx.size(0), self.num_labels)
            onehot.scatter_(1, idx.cpu().long(), 1)
        return onehot.cuda()

    def encode(self, x, c):
        if self.condition:
            c = self.idx2oneHot(c)
            x = torch.cat((x, c), dim=1)
        x = self.features(x)
        x = x.view(-1, self.reside_channel * self.reside_size)
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
        features = self.latent_to_features(z).view(-1, self.reside_channel, self.reside_size)
        result = self.features_to_img(features)
        return nn.functional.interpolate(result, self.input_dim, mode='linear', align_corners=True)

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparmeterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class ConvVAE(VAE):
    def __init__(self, latent_size, input_channel=4, condition=False, num_labels=2):
        """
        Shape: 816 + 531 = (1347, 4) + num_labels ==> 1349, 4 + one zeros ==> 1350, 4
        """
        super(ConvVAE, self).__init__(latent_size, input_channel=input_channel, 
            condition=condition, num_labels=num_labels)
        
        self.input_dim = 1347
        self.hidden_dim = 2048
        self.reside_size = 21
        self.reside_channel = 512
        # Define encoder
        self.features = nn.Sequential(
            # 672
            nn.Conv1d(input_channel + num_labels, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(672),

            # 336
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            # 168
            nn.Conv1d(64, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),

            # 84
            nn.Conv1d(128, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),

            # 42
            nn.Conv1d(256, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),

            # 21
            nn.Conv1d(512, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),
        )
        self.features_to_hidden = nn.Sequential(
            nn.Linear(self.reside_channel * self.reside_size, self.hidden_dim),
            nn.ReLU()
        )
        self.fc2mu = nn.Linear(self.hidden_dim, latent_size)
        self.fc2Logvar = nn.Linear(self.hidden_dim, latent_size)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_size + num_labels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.reside_channel * self.reside_size),
            nn.ReLU()
        )
        # Decoder
        self.features_to_img = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 2, 2),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.ConvTranspose1d(512, 512, 2, 2),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.ConvTranspose1d(256, 256, 2, 2),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 128, 2, 2),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 64, 2, 2),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(32),

            nn.ConvTranspose1d(32, 32, 2, 2),
            nn.Conv1d(32, input_channel, 3, padding=1),
            nn.BatchNorm1d(input_channel),
        )
        self.init_weight()
