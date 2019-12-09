# data_create.py is used to create dataset contain convergenced data and disconvergenced data based on IEEE-CASE.
# The way to create dataset is based on 电科院, detail in: https://www.yuque.com/yas4z2/zr4l99/yubgqy
from pypower.api import case39, ppoption
from custom_runpf import runpf
import numpy as np
import random
import pickle


def read_data():
    with open('ieee_data.pkl', 'rb') as fp:
        data = pickle.load(fp)
    

def runpf_data(data, ppc, ppopt):
    data['baseMVA'] = ppc['baseMVA']
    data['branch'] = ppc['branch']
    success, _ = runpf(data, ppopt)
    return success

def create_data():
    ppc = case39()
    ppopt = ppoption(PF_ALG=1, VERBOSE=0)
    N = 7
    M = 5
    epoch = 50
    random.seed(7)

    bus_number = ppc['bus'].shape[0]
    gen_number = ppc['gen'].shape[0]
    result = {
        'gen': [],
        'bus': [],
        'success': []
    }
    for i in range(epoch):
        print('epoch: ${}'.format(i))
        random_bus = random.sample(range(0, bus_number), N)
        for reinforce in range(1, 41):
            strength = reinforce * 0.1
            sample_data = {
                'bus': ppc['bus'].copy(),
                'gen': ppc['gen'].copy()
            }
            pg_strength = sum(strength * ppc['bus'][random_bus][:,  2])
            qg_strength = sum(strength * ppc['bus'][random_bus][:, 3])
            for index in random_bus:
                sample_data['bus'][index, 2:4] *= (1+strength)
            for gen_idx in range(5):
                random_gen = random.sample(range(0, gen_number), M)
                for index in random_gen:
                    sample_data['gen'][index, 1] += pg_strength / M
                    sample_data['gen'][index, 2] += qg_strength / M
                result['gen'].append(sample_data['gen'].copy())
                result['bus'].append(sample_data['bus'].copy())
                result['success'].append(runpf_data(sample_data, ppc, ppopt))

    print('{}/{}'.format(sum(result['success']), len(result['success'])))
    with open('ieee_data.pkl', 'wb') as fp:
        pickle.dump(result, fp)

