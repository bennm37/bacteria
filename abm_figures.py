from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from default_parameters import *
from pyramid import offset_pyramid
import fdiff_test

def characteristic_trajectory(seed):
    def state_ODE(C,x,y):
        return np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T
    def chemical(x,y):
        return 8*offset_pyramid(x,y,c=[12,6])
    parameters["N"] = int(10)
    parameters["M"] = 100000
    parameters["seed"] = seed
    parameters["initial_condition"] = "delta"
    parameters["Lx"] = 10
    parameters["Ly"] = 10
    parameters["chemical"] = lambda x,y: parabola(x,y)
    parameters["overwrite"] = True
    parameters["save_frequency"] = 10
    parameters["save"] = "all"
    foldername = f"data/N={parameters['N']}_params_6.1_preview"
    parameters["foldername"] = f"data/N={parameters['N']}_params_6.1_preview"
    c = Colony(parameters,False)
    c.run()
    print("Ran simulation")
    a = Analysis(foldername,parameters,verbose=False,stride=1)
    ax = a.plot_trajectory(0,1)
    plt.savefig(f"media/cts/characteristic_trajectory_single_{seed}.png")

def run_figure_6_1():
    pass

def plot_figure_6_1():
    # ABM PLOT
    foldername = "data/figure_6.1_final_data"
    parameters["foldername"] = foldername
    parameters["N"] = int(1e6)
    parameters["Lx"] = 20
    parameters["Ly"] = 10
    a = Analysis(foldername,parameters,verbose=False,stride=1)
    fig = plt.figure()
    ax0 = fig.add_subplot(221,projection="3d")
    ax1 = fig.add_subplot(222,projection="3d")
    ax2 = fig.add_subplot(223,projection="3d")
    ax3 = fig.add_subplot(224,projection="3d")
    a.plot_density(9,nedges=40,scale=(39**2)/1e6,zmax=15,ax=ax0)
    a.plot_density(99,nedges=40,scale=(39**2)/1e6,zmax=9,ax=ax2)
    ax0.set(xlabel='x', ylabel='y', zlabel='density of cells',title="Stochastic model")
    ax0.azim = 225
    ax2.set(xlabel='x', ylabel='y', zlabel='density of cells')
    ax2.azim = 225

    # PDE PLOT
    t,x,y,U,S = fdiff_test.load("data/test_fdiff_k_0_full_final")
    x = np.linspace(0,20,80)
    y = np.linspace(0,10,40)
    X,Y = np.meshgrid(x,y,indexing='ij')
    U = fdiff_test.normalise(U,x,y)*5
    ax1.plot_surface(X,Y,U[20],cmap='coolwarm')
    ax1.set(xlabel='x', ylabel='y', zlabel='density of cells',title="PDE model")
    ax1.set(zlim=(0,15))
    ax1.azim = 225
    ax3.plot_surface(X,Y,U[-1],cmap='coolwarm')
    ax3.set(xlabel='x', ylabel='y', zlabel='density of cells')
    ax3.set(zlim=(0,9))
    ax3.azim = 225
    plt.savefig("media/figure_6.1_redo.pdf")
    # plt.show()


# plot_figure_6_1()