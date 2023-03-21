from pde_solver import PDE_Solver
import numpy as np
import matplotlib.pyplot as plt
from default_parameters import parameters
bump = lambda x,y: np.exp(-((x-0.5)**2+(y-0.5)**2)/0.05)
cosine = lambda x,y: np.cos(np.pi*y)*np.cos(np.pi*x)
delta = lambda x,y : np.where(np.logical_and(x>0.6,x<0.7),np.where(np.logical_and(y>0.6,y<0.7),1,0),0)
def pyramid(x, y): 
    X,Y = 2*x-1, 2*y-1
    pyramid = np.where(np.logical_and(Y<=X,Y<=-X),1+Y,X)
    pyramid = np.where(np.logical_and(Y<=X,Y>=-X),1-X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y<=-X),1+X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y>=-X),1-Y,pyramid)
    return 0.5*pyramid
parameters = {
    "M": 2000,
    "N": 5000,
    "L": 1,
    "te": 1,
    "ta": 1,
    "dt": 0.000008,
    "dx": 0.01,
    # "chemical": lambda x,y: (1*np.ones_like(x)*np.linspace(0,1,x.shape[0])),
    # "chemical": lambda x,y:-50*((x-0.5)**2+(y-0.5)**2),
    "chemical" : pyramid,
    "taxis_strength": 10,
    "n_states":2,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T,
    "lambda0": 5,
    "initial_condition": delta,
    # "initial_condition": lambda x,y: np.ones_like(x),
    # "initial_condition": lambda x,y: ((x-0.5)**2+(y-0.5)**2),
    # "initial_condition": cosine,
    "dimension": 2,
    "seed": 3
}

ps = PDE_Solver(parameters,verbose=True)
ps.explicit_solve()
fig,ax,anim = ps.animate_3d(stride=100)
ax.set(xlabel='x', ylabel='y', zlabel='z')
anim.save("media/dx_0.01_dt_0.0000001.mp4", fps=30)
# checking integrates to constant value over time 
total_density = np.sum(ps.U,axis=(1,2))
fig,ax = plt.subplots()
ax.plot(total_density)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set(xlabel='x', ylabel='y', zlabel='z')
ax.plot_surface(ps.X, ps.Y, ps.initial_condition(ps.X,ps.Y),cmap='coolwarm')
ax1 = fig.add_subplot(122, projection='3d')
ax1.set(xlabel='x', ylabel='y', zlabel='z')
ax1.plot_surface(ps.X, ps.Y, ps.chemical(ps.X,ps.Y),cmap='coolwarm')
plt.show()