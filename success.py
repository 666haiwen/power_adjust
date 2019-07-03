import torch
import numpy as np
from replayMemory import ReplayMemory, Transition
from TrendData import TrendData

successMemory = ReplayMemory(1000, 'memory/success.pkl', True)
successMemory.read()
trendData = TrendData('template/36nodes/')
state = torch.squeeze(successMemory.memory[0].state).cpu().numpy()
trendData.set_state(state)
print(trendData.reward())

