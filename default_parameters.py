import numpy as np
te = 1
ta = 1
sins = lambda x,y: np.sin(x-5)*np.sin(y-5)
cone = lambda x,y: -np.sqrt((x-5)**2+(y-5)**2)
parabola = lambda x,y: -(x-5)**2-(y-5)**2
parameters = {
    "M": 5000,
    "N": 5000,
    "L": 10,
    "te": te,
    "ta": ta,
    "dt": 0.01,
    "dx": 0.01,
    "chemical": parabola,
    "taxis_strength": 10,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/te,(C-y)/ta]).T,
    "lambda0": 5,
    "initial_condition": "uniform",
    "dimension": 2,
    "seed": 3
}
