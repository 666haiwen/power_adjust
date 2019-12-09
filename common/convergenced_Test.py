import numpy as np
import pickle as pkl
import torch

from common.dataloaders import get_case39_dataloader
from common.dataloaders import get_case2k_dataloader
from common.model import VAE, ConvVAE
from env.TrendData import TrendData


DongBei_PARAMS = {
    'dataset': 'DongBei_Case',
    'convergenced':{
        'bool': True,
        'content': ['g', 'l']
    },
    'disconvergenced': {
        'bool': True,
        'rule': False,
        'content': ['g', 'l']
    }
}
class Convergenced(object):
    def __init__(self, model, cuda, dataset):
        """
        Init of Convergenced.
        @params:
            model: the model to test
            cuda: CUDA or not, bool
            dataset: the dataset to fit the model, belong to ['case36', 'DongBei_Case']
        """
        self.model = model
        self.dataset = dataset
        if dataset == 'case36':
            self.trend_data_path = 'env/data/case36/1/11'
            self.dataset_path = 'env/data/case36/'
        elif dataset == 'DongBei_Case':
            self.trend_data_path = 'env/data/dongbei_LF-2000/dataset/1/11/'
            self.dataset_path = 'env/data/dongbei_LF-2000/'
        else:
            raise ValueError("params of env test function must belong to \
                ['case36', 'DongBei_Case'], but input {} instead.".format(dataset))
        self.cuda = cuda
        self.model.eval()

    def test(self, test_loader, params=DongBei_PARAMS):
        """
        Test convergenced dataloader by self model
        @params:
            test_loader: data loader of test data, type: torch.utils.data.Dataset
            params: params of test setting, type:dict default:DongBei_PARAMS
        """
        original_success = []
        original_failed = []
        trendData = TrendData(target='jointvae', path=self.trend_data_path)
        with torch.no_grad():
            for index, (data, labels, path) in enumerate(test_loader):
                if self.cuda:
                    data = data.cuda()
                    labels = labels.cuda()
                epoch_fail, epoch_success, _, _ = self._test_iteration(trendData, data, labels, path, params)
                original_success.extend(epoch_success)
                original_failed.extend(epoch_fail)

        if len(original_success) == 0:
            original_success.append(0)
        if len(original_failed) == 0:
            original_failed.append(0)
        print('[{}/{}]  original success rate: {:.2f}%'.format(sum(original_success), len(original_success), \
            sum(original_success) / len(original_success) * 100))
        print('[{}/{}]  original fail rate: {:.2f}%'.format(sum(original_failed), len(original_failed), \
            sum(original_failed) / len(original_failed) * 100))
        print('Result based params:\n {}'.format(params))

    def _test_iteration(self, trendData, data, labels, path, params):
        reverse_labels = 1 - labels
        mu_batch, _ = self.model.encode(data, labels)
        recon_batch = self.model.decode(mu_batch, labels)
    
        reverse_recon_batch = self.model.decode(mu_batch, reverse_labels)
        shape = recon_batch.shape
        original_success = []
        original_failed = []
        res_label = []
        dis_params = params['disconvergenced']
        normal_params = params['convergenced']
        recon_data_list = []
        for idx in range(shape[0]):
            trendData.reset(path[idx], restate=False)
            if labels[idx] == 0 and dis_params['bool'] == True:
                new_data = reverse_recon_batch[idx].cpu().numpy()
                recon_data_list.append(new_data.copy())
                if dis_params['rule'] == True:
                    for alpha_Pg in [0.9, 1.0, 0.95]:
                        for alpha_Qg in [0.9, 1.0, 0.95]:
                            result = trendData.test(new_data, content=dis_params['content'], dataset=params['dataset'],
                                                    balance=True, alpha_Pg=alpha_Pg, alpha_Qg=alpha_Qg)
                            if result == True:
                                break
                        if result == True:
                            break
                else:
                    result = trendData.test(new_data, content=dis_params['content'], dataset=params['dataset'], balance=False)
                original_failed.append(result)
            elif normal_params['bool'] == True:
                result = trendData.test(recon_batch[idx].cpu().numpy(), dataset=params['dataset'], content=normal_params['content'], balance=False)
                recon_data_list.append(recon_batch[idx].cpu().numpy())
                original_success.append(result)
            res_label.append(result)

        print('Disconvergenced {}/{}'.format(sum(original_failed), len(original_failed)))
        print('Convergenced    {}/{}\n'.format(sum(original_success), len(original_success)))
        return original_failed, original_success, res_label, recon_data_list

    def reverse_recon_dataset(self, test_loader, save_path=None, params=DongBei_PARAMS):
        """
        Record the reverse recon dataset contain data, label(original), recon_label, path(path to original data), params('setting to recon')
        @params:
            test_loader: data loader of test data, type: torch.utils.data.Dataset
            save_path: the path of result, default: None,means self.dataset_path + 'reverse_recon_test_dataset.pkl'
            params: params of test setting, type:dict, default:DongBei_PARAMS
        """
        trendData = TrendData(target='jointvae', path=self.trend_data_path)
        if save_path == None:
            save_path = self.dataset_path + 'reverse_recon_test_dataset.pkl'
        with torch.no_grad():
            test_recon = {
                'data': [],
                'label': test_loader.dataset.label,
                'recon_label': [],
                'path': test_loader.dataset.path,
                'params': params
            }
            for index, (data, labels, path) in enumerate(test_loader):
                if self.cuda:
                    data = data.cuda()
                    labels = labels.cuda()
                _, _, new_label, recon_data = self._test_iteration(trendData, data, labels, path, params)
                test_recon['data'] += recon_data
                test_recon['recon_label'] += new_label
            test_recon['data'] = np.array(test_recon['data'], dtype=np.float32)
            test_recon['recon_label'] = np.array(test_recon['label'], dtype=np.float32)
        
        print('Result based params:\n {}'.format(params))
        with open(save_path, 'wb') as fp:
            pkl.dump(test_recon, fp)
                
