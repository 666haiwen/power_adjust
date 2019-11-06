import os
import numpy as np
import torch
from env.TrendData import TrendData
from torch.nn import functional as F
from torchvision.utils import make_grid


class Tester():
    def __init__(self, model, optimizer, use_cuda=False, saving_path=None):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.saving_path = saving_path
        self.trendData = TrendData(target='jointvae', path='env/data/36nodes_new/1/11/')

    def load(self, epoch):
        if not os.path.exists(self.saving_path):
            raise RuntimeError("There is no model to Load in '{}'".format(self.saving_path))
        else:
            checkpoint = torch.load(self.saving_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizet_state_dict'])
            epoch = checkpoint['eopch']
        return epoch            

    def test(self, data_loader):
        """
        Tests the model.
    
        @Paramters:
        data_loader: torch.utils.data.DataLoader
        """
        self.model.eval()
        original_success = []
        original_fail = []
        with torch.no_grad():
            for index, (data, labels, path) in enumerate(data_loader):
                if self.use_cuda:
                    data = data.cuda()
                    labels = labels.cuda()
                epoch_success, epoch_fail = self._test_iteration(data, labels, path)
                original_success.extend(epoch_success)
                original_fail.extend(epoch_fail)
        return sum(original_success), len(original_success), \
            sum(original_fail), len(original_fail)
    

    def _test_iteration(self, data, labels, path):
        recon_batch, latent_dist = self.model(data)
        recon_batch[:, :self.model.fix_dim] = data[:, :self.model.fix_dim]
        shape = recon_batch.shape
        original_success = []
        original_fail = []
        for idx in range(shape[0]):
            self.trendData.reset(path[idx])
            if labels[idx] == 1:
                result = self.trendData.test(recon_batch[idx])
                original_success.append(result)
            else:
                print(latent_dist)
        return original_success, original_fail
