from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from default_parameters import *

parameters["N"] = 1000
parameters["M"] = 2000
parameters["seed"] = 101
parameters["initial_condition"] = "delta"
# flipped_pyramid = lambda x,y: np.array(pyramid(x,y))[::-1,::-1]
# def flipped_pyramid(x,y):
#     arr = np.array(pyramid(x,y)).reshape(x.shape[0],y.shape[0])
#     return pyramid(x,y)[::-1,::-1]
parameters["chemical"] = lambda x,y: 8*pyramid(x,y)

# fig = plt.figure()
# ax = fig.add_subplot(111,projection="3d")
# x = np.linspace(1,5,100)
# X,Y = np.meshgrid(x,x)
# ax.plot_surface(X,Y,parameters["chemical"](X,Y),cmap="coolwarm")
# plt.show()

c = Colony(parameters,False)
foldername = "data/test"
c.run(overwrite=True,array=False,save_frequency=50,foldername=foldername)
print("Ran simulation")
a = Analysis(foldername,parameters,verbose=False,stride=10)
anim = a.animate_dots(1)
anim.save("media/dots.mp4")

# fig = plt.figure()
# ax = fig.add_subplot(111,projection="3d")
# a.plot_density(-1,ax,nedges=41,zmax=20)
anim = a.animate_density(1,nedges=21)
anim.save("media/test_delta.mp4")
# anim = a.animate_dots(50)
# anim.save()

# anim.save(f"media/taxis_strength_{parameters['taxis_strength']}_animation.mp4")
# plt.savefig(f"media/taxis_strength_{parameters['taxis_strength']}.png")
# anim.save("media/test.mp4")