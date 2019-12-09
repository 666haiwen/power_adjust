"""
    Re-implement of vae example in pytorch-examples.
    This is an improved implementation of the paper [Stochastic Gradient VB and the
    Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
    It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad.
    These changes make the network converge much faster.
"""
import os
import argparse
import time
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from common.dataloaders import get_case39_dataloader
from common.dataloaders import get_case2k_dataloader
from common.model import VAE, ConvVAE
from env.TrendData import TrendData


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IDX = 0
DATA_PATH = ['env/data/case36/1/11', 'env/data/dongbei_LF-2000/dataset/1/11/']
PATH = ['model/case39_cvae', 'model/case2K_cvae']
def get_args():
    parser = argparse.ArgumentParser(description='VAE MINST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size of trainning (default = 128)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs to train (default = 1500)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA traning or not (default = True)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='num of workers while training and testing (default = 2)')
    parser.add_argument('--path', type=str, default=PATH[IDX],
                        help='path to model saving')
    parser.add_argument('--data-path', type=str, default=DATA_PATH[IDX],
                        help='path to model saving')
    parser.add_argument('--load-checkpoint', type=bool, default=True,
                        help='load history model or not (default = True)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate of training (default = 1e-4)')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sample', type=bool, default=False,
                        help='Test to get sample img or not')
    parser.add_argument('--latent-size', type=int, default=128,
                        help='number of latents (default = 128)')
    parser.add_argument('--conditional',  type=bool, default=True)
    parser.add_argument('--beta', type=float, default=1.0, help='weight of loads mse')
    args = parser.parse_args()
    args.path = args.path + '_{}.pth'.format(args.latent_size)
    args.cuda = args.cuda and torch.cuda.is_available()
    return args


def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed 0.5 by lr_update_epoch"""
    lr = max(1e-7, lr / np.sqrt(2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr, optimizer

def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # KL_Distance : 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE - KLD

def train(model, optimizer, train_loader, epoch_begin, lr):
    model.train()
    lr = lr
    history_loss_min = 1e10
    history_loss_epoch = 0
    for epoch in range(epoch_begin + 1, args.epochs):
        before_time = time.time()
        train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs, labels)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.3f}\tLoss: {:.3f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(inputs), loss.item()))
        # finish one epoch
        if train_loss < history_loss_min:
            history_loss_min = train_loss
            history_loss_epoch = epoch
        if epoch - history_loss_epoch >= 15:
            lr, optimizer = adjust_learning_rate(lr, optimizer)
            history_loss_epoch = epoch
            print('--------Update Lr[{:.7f}]-------'.format(lr))
        # log
        time_cost = time.time() - before_time
        print('====> Epoch: {} \tAverage loss : {:.3f}\tTime cost: {:.0f}\tLR: {:.7f}'.format(
            epoch, train_loss / len(train_loader.dataset), time_cost, lr))
        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr': lr
            }, args.path)


def main():
    if IDX == 0:
        model = VAE(args.latent_size, input_channel=2, condition=args.conditional, num_labels=2)
    elif IDX == 1:
        model = ConvVAE(args.latent_size, input_channel=4, condition=args.conditional, num_labels=2)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.path = os.path.join(os.getcwd(), args.path)
    epoch = -1
    lr = args.lr
    if args.load_checkpoint:
        if os.path.exists(args.path):
            checkpoint = torch.load(args.path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr = checkpoint['lr']
        else:
            print('Doesn\'t find checkpoint in ' + args.path)
    print('Begin train.......')
    train(model, optimizer, train_loader, epoch, lr)
    print('Finish train!')


if __name__ == "__main__":
    args = get_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda \
    else {'num_workers': args.num_workers}

    if IDX == 1:
        train_loader = get_case2k_dataloader(batch_size=args.batch_size)
        test_loader = get_case2k_dataloader(batch_size=args.batch_size, test=True)
    elif IDX == 0:
        train_loader = get_case39_dataloader(batch_size=args.batch_size)
        test_loader = get_case39_dataloader(batch_size=args.batch_size, test=True)

    loads_num = [10, 816][IDX]
    generators_num = [9, 531][IDX]
    main()
