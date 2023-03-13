from solver import Colony,Analysis
import matplotlib.pyplot as plt
import numpy as np
te = 1
ta = 1
parameters = {
    "M": 5000,
    "dt": 0.01,
    "N": 10000,
    "L": 10,
    "chemical": lambda x,y: 1-(x-5)**2-(y-5)**2,
    "taxis_strength": 1,
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
a.plot_density(-1)
# anim = a.animate_dots(50)
# anim.save("test.mp4")