from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from default_parameters import *

parameters["N"] = 1000
parameters["M"] = 5000
parameters
c = Colony(parameters,False)
foldername = "data/test"
c.run(overwrite=True,array=False,save_frequency=50,foldername=foldername)
print("Ran simulation")
a = Analysis(foldername,parameters,verbose=False)
# fig = plt.figure()
# ax = fig.add_subplot(111,projection="3d")
# a.plot_density(-1,ax,nedges=41,zmax=20)
anim = a.animate_density(1,nedges=21)
anim.save("media/test_loading.mp4",fps=10)
# anim = a.animate_dots(50)


# anim.save(f"media/taxis_strength_{parameters['taxis_strength']}_animation.mp4")
# plt.savefig(f"media/taxis_strength_{parameters['taxis_strength']}.png")
# anim.save("media/test.mp4")