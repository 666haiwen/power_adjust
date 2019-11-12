import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader


def get_case39_dataloader(batch_size=512,
                            test=False,
                            transform=True,
                            path_to_data='env/data/36nodes_new/train.pkl'):
    """DSprites dataloader."""
    case39_data = Case39Dataset(path_to_data, test=test, transform=transform)
    case39_loader = DataLoader(case39_data, batch_size=batch_size,
                                 shuffle=True)
    return case39_loader


class Case39Dataset(Dataset):
    """Case39 dataset.
        Data shape (numbers, 2):
            (pg, qg) of generators
            (pg, qg) of loads
            (mark, mark) of acs
    """
    def __init__(self, path_to_data, test=False, transform=True):
        """
        Parameters
        ----------
        path_to_data : str
            path to the dataset
        """
        self.test = test
        self.transform = transform
        with open(path_to_data, 'rb') as fp:
            self.dataset = pkl.load(fp)
            self.data = self.dataset['data']
            self.label = self.dataset['label']
            if test:
                self.path = self.dataset['path']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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