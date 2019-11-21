import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_case39_dataloader(batch_size=512,
                            test=False,
                            path_to_data='env/data/36nodes_new/train.pkl',
                            gan=False,
                            num_workers=1):
    """DSprites dataloader."""
    if test:
        path_to_data = 'env/data/36nodes_new/test.pkl'
    case39_data = Case39Dataset(path_to_data, test=test, gan=gan)
    case39_loader = DataLoader(case39_data, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    return case39_loader


class Case39Dataset(Dataset):
    """Case39 dataset.
        Data shape (numbers, 2):
            (pg, qg) of loads
            (pg, qg) of generators
            (mark, mark) of acs
    """
    def __init__(self, path_to_data, test=False, gan=False):
        """
        Parameters
        ----------
        path_to_data : str
            path to the dataset
        """
        self.test = test
        self.gan = gan
        with open(path_to_data, 'rb') as fp:
            self.dataset = pkl.load(fp)
            self.data = self.dataset['data']
            self.label = self.dataset['label']
            if test:
                self.path = self.dataset['path']
        if gan:
            self.convergenced_items = []
            self.disconvergenced_items = []
            for i, v in enumerate(self.label):
                if v == 1:
                    self.convergenced_items.append(i)
                else:
                    self.disconvergenced_items.append(i)
            self.convergenced_len = len(self.convergenced_items)
            self.disconvergenced_len = len(self.disconvergenced_items)

    def __len__(self):
        if not self.gan:
            return len(self.data)
        else:
            return max(self.convergenced_len, self.disconvergenced_len)

    def __getitem__(self, idx):
        if not self.gan:
            sample = self.data[idx]
            label = self.label[idx]

            # Add extra dimension to turn shape into (H, W) -> (H, W, C)
            # sample = sample.reshape((1,) + sample.shape)
            sample = torch.from_numpy(sample).float()
            if not self.test:
                return sample, label
            else:
                path = self.path[idx]
                return sample, label, path
        else:
            convergenced_idx = idx % self.convergenced_len
            convergenced_sample = self.data[self.convergenced_items[convergenced_idx]]

            disconvergenced_idx = idx % self.disconvergenced_len
            disconvergenced_sample = self.data[self.disconvergenced_items[disconvergenced_idx]]

            convergenced_sample = torch.from_numpy(convergenced_sample).float()
            disconvergenced_sample = torch.from_numpy(disconvergenced_sample).float()
            
            return {'A':convergenced_sample, 'B':disconvergenced_sample}


def get_case2k_dataloader(batch_size=512,
                            test=False,
                            path_to_data='env/data/dongbei_LF-2000/train.pkl',
                            num_workers=1):
    """DSprites dataloader."""
    if test:
        path_to_data = 'env/data/dongbei_LF-2000/test.pkl'
    case2k_data = Case2KDataset(path_to_data, test=test)
    case2k_loader = DataLoader(case2k_data, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    return case2k_loader


class Case2KDataset(Dataset):
    """Case39 dataset.
        Data shape (numbers, 4):
            (mark, pg, qg, vBase) of loads
            (mark, pg, qg, vBase) of generators
        numbers: 1347
    """
    def __init__(self, path_to_data, test=False):
        """
        Parameters
        ----------
        path_to_data : str
            path to the dataset
        """
        self.test = test
        with open(path_to_data, 'rb') as fp:
            self.dataset = pkl.load(fp)
            self.data = self.dataset['data']
            self.label = self.dataset['label']
            if test:
                self.path = self.dataset['path']
        self.data = self.data.swapaxes(1, 2)
        # shape = self.data.shape
        # new_data = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
        # self.data = np.concatenate((self.data, new_data), axis=-1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label[idx]

        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        # sample = sample.reshape((1,) + sample.shape)
        sample = torch.from_numpy(sample).float()
        # sample = torch.sigmoid(torch.from_numpy(sample).float())
        if not self.test:
            return sample, label
        else:
            path = self.path[idx]
            return sample, label, path
