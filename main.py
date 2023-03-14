from solver import Colony,Analysis
import matplotlib.pyplot as plt
import numpy as np
from default_parameters import *
c = Colony(parameters,False)
location_data,state_data,velocity_data,signal_data = c.run()
print("Ran simulation")
a = Analysis(location_data,state_data,velocity_data,parameters)
# anim = a.animate_density(50)
# plt.show()
# anim.save(f"media/taxis_strength_{parameters['taxis_strength']}_animation.mp4")
# plt.savefig(f"media/taxis_strength_{parameters['taxis_strength']}.png")
anim = a.animate_dots(50)
plt.show()
# anim.save("media/test.mp4")