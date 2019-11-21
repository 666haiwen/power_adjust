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
from model import VAE, ConvVAE
from env.TrendData import TrendData


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IDX = 1
DATA_PATH = ['env/data/36nodes_new/1/11', 'env/data/dongbei_LF-2000/dataset/1/11/']
PATH = ['model/case39_cvae', 'model/case2K_cvae']
CONTENT = [['g', 'ac'], ['g']]
def get_args():
    parser = argparse.ArgumentParser(description='VAE MINST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size of trainning (default = 128)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA traning or not (default = True)')
    parser.add_argument('--path', type=str, default=PATH[IDX],
                        help='path to model saving')
    parser.add_argument('--data-path', type=str, default=DATA_PATH[IDX],
                        help='path to model saving')
    parser.add_argument('--load-checkpoint', type=bool, default=True,
                        help='load history model or not (default = True)')
    args = parser.parse_args()
    args.path = args.path + '_enhanced_loads_beta{:.1f}_{}.pth'.format(args.beta, args.latent_size)
    args.cuda = args.cuda and torch.cuda.is_available()
    return args

def test(model, test_loader):
    model.eval()
    original_success = []
    original_failed = []
    trendData = TrendData(target='jointvae', path=args.data_path)
    with torch.no_grad():
        for index, (data, labels, path) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
                labels = labels.cuda()
            epoch_fail, epoch_success = _test_iteration(model, trendData, data, labels, path)
            original_success.extend(epoch_success)
            original_failed.extend(epoch_fail)
    if len(original_success) == 0:
        original_success.append(0)
    print('[{}/{}]  original success rate: {:.2f}%'.format(sum(original_success), len(original_success), \
        sum(original_success) / len(original_success) * 100))
    print('[{}/{}]  original fail rate: {:.2f}%'.format(sum(original_failed), len(original_failed), \
        sum(original_failed) / len(original_failed) * 100))

def _test_iteration(model, trendData, data, labels, path):
    reverse_labels = 1 - labels
    mu_batch, _ = model.encode(data, labels)
    recon_batch = model.decode(mu_batch, labels)

    reverse_recon_batch = model.decode(mu_batch, reverse_labels)
    shape = recon_batch.shape
    original_success = []
    original_failed = []
    for idx in range(shape[0]):
        trendData.reset(path[idx], restate=False)
        if labels[idx] == 0:
            # continue
            new_data = reverse_recon_batch[idx].cpu().numpy()
            for alpha in [0.906]:
                result = trendData.test(new_data, content=CONTENT[IDX], balance=True, alpha=alpha)
                if result == True:
                    break
            original_failed.append(result)
        else:
            continue
            result = trendData.test(recon_batch[idx].cpu().numpy(), content=['g', 'l'], balance=True, alpha=1.05)
            original_success.append(result)
        print('Disconvergenced {}/{}'.format(sum(original_failed), len(original_failed)))
        print('Convergenced    {}/{}\n'.format(sum(original_success), len(original_success)))
    return original_failed, original_success

def main():
    if IDX == 0:
        model = VAE(134 + 19 * 2, args.latent_size, args.conditional, 2)
    elif IDX == 1:
        model = ConvVAE(args.latent_size, condition=args.conditional, num_labels=2)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.path = os.path.join(os.getcwd(), args.path)
    if args.load_checkpoint:
        if os.path.exists(args.path):
            checkpoint = torch.load(args.path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('Doesn\'t find checkpoint in ' + args.path)
            return
    print('Finish train!\nBegin test.......')
    test(model, test_loader)


if __name__ == "__main__":
    args = get_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda \
    else {'num_workers': args.num_workers}
    # torch.manual_seed(args.seed)

    if IDX == 1:
        train_loader = get_case2k_dataloader(batch_size=args.batch_size)
        test_loader = get_case2k_dataloader(batch_size=args.batch_size, test=True)
    elif IDX == 0:
        train_loader = get_case39_dataloader(batch_size=args.batch_size)
        test_loader = get_case39_dataloader(batch_size=args.batch_size, test=True)

    # generators_num = 9
    # loads_num = 10

    loads_num = 816
    generators_num = 531
    main()
