from pypower.api import case39, ppoption, printpf, case9
from custom_runpf import runpf
# Simulation modules
ppc = case39()
# ppc['branch'] = ppc['branch'][:-1]
# ppc['gen'][:,1] = 500
ppc['branch'][0][2:] = 0
ppopt = ppoption(PF_ALG=1, VERBOSE=1)
r, success, new_ppc = runpf(ppc, ppopt)
print(new_ppc['bus'][:, 7:10])
# printpf(r)

