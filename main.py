from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from default_parameters import *
from pyramid import offset_pyramid

def state_ODE(C,x,y):
    return np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T
def chemical(x,y):
    return 8*offset_pyramid(x,y,c=[12,6])
parameters["N"] = int(10)
parameters["M"] = 200000
parameters["seed"] = 1
parameters["initial_condition"] = "delta"
parameters["Lx"] = 20
parameters["Ly"] = 10
parameters["chemical"] = lambda x,y: 8*offset_pyramid(x,y,c=[12,6])
parameters["overwrite"] = True
parameters["save_frequency"] = 10
parameters["save"] = "all"
foldername = f"data/N={parameters['N']}_params_6.1_preview"
parameters["foldername"] = f"data/N={parameters['N']}_params_6.1_preview"


c = Colony(parameters,False)

c.run()
print("Ran simulation")
a = Analysis(foldername,parameters,verbose=False,stride=1)
nedges = 20
ax = a.plot_trajectory(0)
# scale = (nedges-1)**2/1e6
# anim = a.animate_density(1,nedges=nedges,scale=scale)
# anim = a.animate_dots()
plt.show()