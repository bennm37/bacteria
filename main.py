from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from default_parameters import *
from pyramid import offset_pyramid

parameters["N"] = int(1e4)
parameters["M"] = 2000
parameters["seed"] = 101
parameters["initial_condition"] = "delta"
parameters["Lx"] = 20
parameters["Ly"] = 10
parameters["chemical"] = lambda x,y: 8*offset_pyramid(x,y,c=[12,6])
parameters["overwrite"] = True
parameters["save_frequency"] = 100
parameters["save"] = "all"


c = Colony(parameters,False)
foldername = f"data/N={parameters['N']}_params_6.1"
c.run()
print("Ran simulation")
a = Analysis(foldername,parameters,verbose=False,stride=1)
anim = a.animate_density(1,nedges=21)
plt.show()