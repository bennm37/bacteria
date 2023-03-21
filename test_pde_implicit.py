from pde_solver import *


parameters = {
    "M": 12000,
    "N": 5000,
    "L": 1,
    "te": 1,
    "ta": 1,
    "dt": 0.0000008,
    "dx": 0.05,
    "chemical" : lambda x: x**2,
    "taxis_strength": 10,
    "n_states":2,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T,
    "lambda0": 5,
    "initial_condition": lambda x: np.where(0.6<x<0.8,1,0),
    "dimension": 1,
    "seed": 0
}
ps = PDE_Solver(parameters)
ps.implicit_solve()
