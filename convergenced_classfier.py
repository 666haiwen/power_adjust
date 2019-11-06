import torch
import argparse
from jointvae.models import VAE
from common.dataloaders import get_case39_dataloader
from model import EasyLinear
from torch import optim


def train(data_loader, epochs):
    for epoch in range(epochs):
        loss_epoch = 0
        accurate_epoch = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda().long()

            optimizer.zero_grad()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            accurate_epoch += torch.sum(preds == labels.data)

        loss_epoch /= len(data_loader.dataset)
        accurate_epoch = accurate_epoch.double()
        accurate_epoch /= len(data_loader.dataset)

        print('epoch: [{}]  Loss: {:.4f} Acc: {:.4f}'.format(epoch, loss_epoch, accurate_epoch))


def test(data_loader):
    model.eval()
    accurate_epoch = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        if use_cuda:
            data = data.cuda()
            labels = labels.cuda().long()

        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        accurate_epoch += torch.sum(preds == labels.data)

    accurate_epoch = accurate_epoch.double()
    accurate_epoch /= len(data_loader.dataset)

    print('Acc: {:.4f}'.format(accurate_epoch))

parser = argparse.ArgumentParser(description='Joinvae of convergenced problem')
parser.add_argument('--train', type=bool, default=True,
                    help='input batch size of trainning (default = 128)')
args = parser.parse_args()

batch_size = 512
lr = 1e-3
epochs = 1000
torch.manual_seed(7)
# Check for cuda
use_cuda = torch.cuda.is_available()


# Define latent spec and model
model = EasyLinear(134 + 19 * 2, use_cuda)

if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

print('Begin train!')
# Load data
data_loader = get_case39_dataloader(batch_size=batch_size, transform=False)
train(data_loader, 50)

print('Begin test!')
data_loader = get_case39_dataloader(path_to_data='env/data/36nodes_new/test.pkl', 
    transform=False, batch_size=batch_size)
test(data_loader)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'model/36nodes-classifer-model.pt')