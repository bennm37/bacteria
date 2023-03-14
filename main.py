from solver import Colony,Analysis
import matplotlib.pyplot as plt
import numpy as np
te = 1
ta = 1
sins = lambda x,y: np.sin(x-5)*np.sin(y-5)
cone = lambda x,y: -np.sqrt((x-5)**2+(y-5)**2)
parabola = lambda x,y: -(x-5)**2-(y-5)**2

parameters = {
    "M": 5000,
    "dt": 0.01,
    "N": 5000,
    "L": 10,
    "chemical": parabola,
    "taxis_strength": 10,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/te,(C-y)/ta]).T,
    "lambda0": 5,
    "initial_condition": "uniform",
    "dimension": 2,
    "seed": 3
}
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