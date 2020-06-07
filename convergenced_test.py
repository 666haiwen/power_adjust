import os
import argparse
import time
import torch
import numpy as np
import pickle as pkl

from common.dataloaders import get_case36_dataloader, get_case2k_dataloader
from common.dataset_create import dataLoader_36Nodes, dataLoader_2000Nodes
from common.model import VAE, ConvVAE
from common.convergenced_Test import Convergenced
from env.TrendData import TrendData


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
PATH = ['model/case36_cvae_32.pth', 'model/case2K_cvae_128.pth', 'model/new_case2K_cvae_128.pth']
RECON_PATH = ['env/data/case36/recon.pkl', 'env/data/dongbei_LF-2000/recon.pkl', 'env/data/case2000/recon.pkl']
CONTENT = [['g', 'ac'], ['g']]

def get_args():
    parser = argparse.ArgumentParser(description='VAE MINST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size of trainning (default = 512)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA traning or not (default = True)')
    parser.add_argument('--path', type=str, default='model/new_case2K_cvae_128.pth',
                        help='path to model saving')
    parser.add_argument('--dataset', type=int, default=2,
                        help="dataset to choose['case36', 'DongBei_Case', 'case2K'], value of index")
    parser.add_argument('--latent-size', type=int, default=128,
                        help='number of latents (default = 128)')
    args = parser.parse_args()
    args.path = PATH[args.dataset]
    args.cuda = args.cuda and torch.cuda.is_available()
    return args

if __name__ == '__main__':
    args = get_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda \
        else {'num_workers': args.num_workers}
    dataset = ''
    if args.dataset == 1 or args.dataset == 2:
        dataset = 'DongBei_Case' if args.dataset == 1 else 'case2000'
        data_loader = dataLoader_2000Nodes()
        model = ConvVAE(args.latent_size, input_channel=4, condition=True, num_labels=2)
        train_loader = get_case2k_dataloader(batch_size=args.batch_size)
        test_loader = get_case2k_dataloader(batch_size=args.batch_size, test=True)
    elif args.dataset == 0:
        dataset = 'case36'
        data_loader = dataLoader_36Nodes()
        model = VAE(args.latent_size, input_channel=2, condition=True, num_labels=2)
        train_loader = get_case36_dataloader(batch_size=args.batch_size)
        test_loader = get_case36_dataloader(batch_size=args.batch_size, test=True)
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.path):
            checkpoint = torch.load(args.path)
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Doesn\'t find checkpoint in ' + args.path)

    convergenced_test = Convergenced(model, args.cuda, dataset, data_loader)
    convergenced_test.test(test_loader, params='case2000')
    # convergenced_test.reverse_recon_dataset(test_loader)
    # convergenced_test.distance_test()

    # with open('env/data/case36/test_latent_dataset.pkl', 'rb') as fp:
    #     dataset = pkl.load(fp)
    # print(111)
