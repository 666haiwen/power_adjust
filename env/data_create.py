from pypower.api import case39, ppoption
from custom_runpf import runpf
import numpy as np
import random
import pickle


def runpf_data(data):
    data['baseMVA'] = ppc['baseMVA']
    data['branch'] = ppc['branch']
    success, _ = runpf(data, ppopt)
    return success

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
    for reinforce in range(40):
        strength = reinforce * 0.1
        sample_data = {
            'bus': ppc['bus'].copy(),
            'gen': ppc['gen'].copy()
        }
        pg_strength = sum(strength * ppc['bus'][random_bus, 2])
        qg_strength = sum(strength * ppc['bus'][random_bus, 3])
        sample_data['bus'][random_bus, 2:4] *= (strength + 1)
        for gen_idx in range(4):
            random_gen = random.sample(range(0, gen_number), M)
            sample_data['gen'][random_gen][1] += pg_strength / M
            sample_data['gen'][random_gen][2] += qg_strength / M
            result['gen'].append(sample_data['gen'].copy())
            result['bus'].append(sample_data['bus'].copy())
            result['success'].append(runpf_data(sample_data))

print(sum(result['success']))
with open('ieee_data.pkl') as fp:
    pickle.dump(result, fp)