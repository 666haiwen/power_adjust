#!/usr/bin/python3

import os
import shutil
import argparse
import time
import itertools
import torch
import numpy as np

from torch import nn, optim
from tensorboardX import SummaryWriter

from env.TrendData import TrendData
from cyclegan.models import Generator, Discriminator
from cyclegan.utils import ReplayBuffer, LambdaLR, weights_init_normal
from common.dataloaders import get_case36_dataloader


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed (default = 7)')
    parser.add_argument('--batch-size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--path', type=str, default='model/case39_cycleGan_tar.pth',
                    help='path to model saving')
    parser.add_argument('--load-checkpoint', type=bool, default=True,
                        help='load history model or not (default = True)')
    parser.add_argument('--input_dim', type=int, default=172, help='number of dims of input data')
    parser.add_argument('--output_dim', type=int, default=172, help='number of dims of output data')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA traning or not (default = True)')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. \
        Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. \
        For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss,\
        please set lambda_identity = 0.1')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    return args


def train(train_loader, epoch_begin):
    ###### Training ######
    lambda_idt = args.lambda_identity
    lambda_A = args.lambda_A
    lambda_B = args.lambda_B
    pos_labels = torch.FloatTensor([1] * args.batch_size).cuda()
    neg_labels = torch.FloatTensor([0] * args.batch_size).cuda()

    cnt = 0
    for epoch in range(epoch_begin + 1, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Set model input
            real_A = batch['A'].cuda()
            real_B = batch['B'].cuda()

            fake_B = netG_A2B(real_A)
            rec_A = netG_B2A(fake_B)
            fake_A = netG_B2A(real_B)
            rec_B = netG_A2B(fake_A)

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            idt_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(idt_B, real_B) * lambda_B * lambda_idt
            # G_B2A(A) should equal A if real A is fed
            idt_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(idt_A, real_A) * lambda_B * lambda_idt
            # GAN loss
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, pos_labels)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, pos_labels)
            # Cycle loss
            loss_cycle_ABA = criterion_cycle(rec_A, real_A) * lambda_A
            loss_cycle_BAB = criterion_cycle(rec_B, real_B) * lambda_B
            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, pos_labels)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, neg_labels)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, pos_labels)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, neg_labels)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            writer.add_scalars('loss', {
                'loss_G': loss_G, 
                'loss_G_identity': (loss_identity_A + loss_identity_B), 
                'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                'loss_D': (loss_D_A + loss_D_B)
            }, cnt)
            cnt += 1

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        print('====> Epoch: {} \tLoss G: {:.4f}\n\
            Loss G identity: {:.4f}\tLoss G GAN: {:.4f}\n\
            Loss G Cycle: {:.4f}\tLoss D: {:.4f}'.format(
            epoch, loss_G, loss_identity_A + loss_identity_B, loss_GAN_A2B + loss_GAN_B2A,
            loss_cycle_ABA + loss_cycle_BAB, loss_D_A + loss_D_B))
        # Save models checkpoints
        if epoch % 20 == 0 or epoch + 1 == args.epochs:
            torch.save({
                'epoch': epoch,
                'netG_A2B_state_dict': netG_A2B.state_dict(),
                'netG_B2A_state_dict': netG_B2A.state_dict(),
                'netD_A_state_dict': netD_A.state_dict(),
                'netD_B_state_dict': netD_B.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
                'optimizer_D_B_state_dict': optimizer_D_B.state_dict()
            }, args.path)

if __name__ == "__main__":
    ###### Definition of variables ######
    args = get_args()
    torch.manual_seed(args.seed)
    
    # Networks
    netG_A2B = Generator(args.input_dim, args.output_dim)
    netG_B2A = Generator(args.output_dim, args.input_dim)
    netD_A = Discriminator(args.input_dim)
    netD_B = Discriminator(args.output_dim)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    if args.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if args.load_checkpoint:
        if os.path.exists(args.path):
            checkpoint = torch.load(args.path)
            netG_A2B.load_state_dict(checkpoint['netG_A2B_state_dict'])
            netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])
            netD_A.load_state_dict(checkpoint['netD_A_state_dict'])
            netD_B.load_state_dict(checkpoint['netD_B_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
            optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
            epoch = checkpoint['epoch']
        else:
            epoch = 0
            print('Doesn\'t find checkpoint in ' + args.path)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, epoch, args.decay_epoch).step)

    # Inputs & targets memory allocation
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    train_loader = get_case36_dataloader(batch_size=args.batch_size, transform=False, 
        gan=True, num_workers=args.n_cpu)
    test_loader = get_case36_dataloader(transform=False, batch_size=args.batch_size, 
        test=True, num_workers=args.n_cpu)
    
    # Train
    print('Begin to train!')
    if os.path.exists('log/cycle-Gan'):
        shutil.rmtree('log/cycle-Gan')
    os.mkdir('log/cycle-Gan')
    writer = SummaryWriter('log/cycle-Gan/')
    train(train_loader, epoch)

