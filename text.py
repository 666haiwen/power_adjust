import torch
import numpy as np

a = np.zeros((2,3,4))
b = torch.tensor(a)
print(a.shape)
c = a.transpose((2,0,1))
print(c.shape)
d = b.permute(2,0,1)
print(d.shape)