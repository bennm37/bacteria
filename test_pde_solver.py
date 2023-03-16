from pde_solver import PDE_Solver
import numpy as np
import matplotlib.pyplot as plt
from default_parameters import parameters
bump = lambda x,y: np.exp(-((x-0.5)**2+(y-0.5)**2)/0.05)
cosine = lambda x,y: np.cos(np.pi*y)*np.cos(np.pi*x)
parameters = {
    "M": 5000,
    "N": 5000,
    "L": 1,
    "te": 1,
    "ta": 1,
    "dt": 0.00001,
    "dx": 0.1,
    "chemical": lambda x,y: (np.ones_like(x)*np.linspace(0,1,x.shape[0])),
    # "chemical": lambda x,y:-50*((x-0.5)**2+(y-0.5)**2),
    "taxis_strength": 10,
    "n_states":2,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T,
    "lambda0": 5,
    "initial_condition": lambda x,y: np.ones_like(x)*np.exp(np.linspace(0,1,x.shape[0])),
    # "initial_condition": lambda x,y: ((x-0.5)**2+(y-0.5)**2),
    # "initial_condition": cosine,
    "dimension": 2,
    "seed": 3
}

ps = PDE_Solver(parameters,verbose=True)
ps.explicit_solve()
fig,ax,anim = ps.animate_3d(stride=1)
ax.set(xlabel='x', ylabel='y', zlabel='z')
plt.show()
# checking integrates to constant value over time 
total_density = np.sum(ps.U,axis=(1,2))
plt.plot(total_density)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set(xlabel='x', ylabel='y', zlabel='z')
ax.plot_surface(ps.X, ps.Y, ps.initial_condition(ps.X,ps.Y),cmap='coolwarm')
ax1 = fig.add_subplot(122, projection='3d')
ax1.set(xlabel='x', ylabel='y', zlabel='z')
ax1.plot_surface(ps.X, ps.Y, ps.chemical(ps.X,ps.Y),cmap='coolwarm')
plt.show()