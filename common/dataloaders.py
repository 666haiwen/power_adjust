import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader


def get_case39_dataloader(batch_size=512,
                            test=False,
                            transform=True,
                            path_to_data='env/data/36nodes_new/train.pkl',
                            gan=False,
                            num_workers=1):
    """DSprites dataloader."""
    if test:
        path_to_data = 'env/data/36nodes_new/test.pkl'
    case39_data = Case39Dataset(path_to_data, test=test, transform=transform, gan=gan)
    case39_loader = DataLoader(case39_data, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
    return case39_loader


class Case39Dataset(Dataset):
    """Case39 dataset.
        Data shape (numbers, 2):
            (pg, qg) of generators
            (pg, qg) of loads
            (mark, mark) of acs
    """
    def __init__(self, path_to_data, test=False, transform=True, gan=False):
        """
        Parameters
        ----------
        path_to_data : str
            path to the dataset
        """
        self.test = test
        self.transform = transform
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
            if self.transform:
                sample[:38] = torch.sigmoid(sample[:38])
            # sample = torch.sigmoid(torch.from_numpy(sample).float())
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
            if self.transform:
                convergenced_sample[:38] = torch.sigmoid(convergenced_sample[:38])
                disconvergenced_sample[:38] = torch.sigmoid(disconvergenced_sample[:38])
            
            return {'A':convergenced_sample, 'B':disconvergenced_sample}
