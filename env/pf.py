from pypower.api import case39, newtonpf, runpf, ppoption, printpf, case300

# Dynamic model classes
from pydyn.controller import controller
from pydyn.sym_order6a import sym_order6a
from pydyn.sym_order4 import sym_order4
from pydyn.ext_grid import ext_grid

# Simulation modules
print('!!!!!')
ppc = case39()
ppopt = ppoption(PF_ALG=2)
r = runpf(ppc, ppopt)

# printpf(r)