from pypower.api import case39, ppoption
from custom_runpf import runpf
# Simulation modules
ppc = case39()
ppc['gen'][:,1] -= 250
ppopt = ppoption(PF_ALG=1, VERBOSE=0)
r = runpf(ppc, ppopt)
print(r)
# printpf(r)