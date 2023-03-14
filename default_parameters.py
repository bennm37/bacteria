import numpy as np
sins = lambda x,y: np.sin(x-5)*np.sin(y-5)
cone = lambda x,y: -np.sqrt((x-5)**2+(y-5)**2)
parabola = lambda x,y: -(x-5)**2-(y-5)**2
parameters = {
    "M": 5000,
    "N": 5000,
    "L": 10,
    "te": 1,
    "ta": 1,
    "dt": 0.01,
    "dx": 0.01,
    "chemical": lambda x,y: -(x-5)**2-(y-5)**2,
    "taxis_strength": 10,
    "n_states":2,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T,
    "lambda0": 5,
    "initial_condition": "uniform",
    "dimension": 2,
    "seed": 3
}
