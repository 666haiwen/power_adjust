import torch
import argparse
from jointvae.models import VAE
from jointvae.training import Trainer
from jointvae.testing import Tester
from common.dataloaders import get_case39_dataloader
from torch import optim



parser = argparse.ArgumentParser(description='Joinvae of convergenced problem')
parser.add_argument('--train', type=bool, default=True,
                    help='input batch size of trainning (default = 128)')
args = parser.parse_args()

batch_size = 512
lr = 1e-3
epochs = 10000

# Check for cuda
use_cuda = torch.cuda.is_available()


# Define latent spec and model
latent_spec = {'cont': 10, 'disc': [2]}
model = VAE(dim=134 + 19 * 2, fix_dim=19 * 2, latent_spec=latent_spec,
            use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
if args.train:
    # Load data
    data_loader = get_case39_dataloader(batch_size=batch_size)
    trainer = Trainer(model, optimizer, lr,
                    cont_capacity=[0.0, 5.0, 25000, 30],
                    disc_capacity=[0.0, 5.0, 25000, 30],
                    use_cuda=use_cuda,
                    saving_path='model/36nodes-vae-model.pt')

    # Train model for 100 epochs
    trainer.train(data_loader, epochs)

    # Save trained model
    trainer.save(epochs)
    # torch.save(trainer.model.state_dict(), 'model/36nodes-vae-model.pt')

## TEST!
else:
    data_loader = get_case39_dataloader(path_to_data='env/data/36nodes_new/test.pkl', 
            test=True,  batch_size=batch_size)
    tester = Tester(model, optimizer, use_cuda=use_cuda, 
            saving_path='model/36nodes-vae-model.pt')
    tester.test(data_loader)