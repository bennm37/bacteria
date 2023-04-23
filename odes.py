import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
font = {'weight' : 'normal',
        'size'   : 14}

mpl.rc('font', **font)
# analysis of ODEs for multiple signals 
def hill(x,K,n):
    return x**n/(K**n+x**n)
# plot the hill function in different colours for
# different values of n
def plot_hill():
    cmap = mpl.colormaps["gist_heat"]
    x = np.linspace(0,3,1000)
    fig,ax = plt.subplots()
    for n in [0.25,0.5,1,2,4,8]:
        color = cmap(np.log(n)/np.log(16))
        ax.plot(x,hill(x,1,n),label=f"n={n}",c=color)
    ax.vlines(1,0,0.5,linestyles="dashed",color="k")
    ax.hlines(0.5,0,1,linestyles="dashed",color="k")
    ax.set(xlabel="Free Ligands",ylabel="Fraction of Ligands Bound",title="Hill functions")
    ax.set(ylim=(0,1.1))
    ax.legend()
    plt.tight_layout()
    plt.savefig("media/hill_functions.png")

def step(t,c):
    return np.heaviside(t-c,1)

def plot_step(a,b,c):
    t = np.linspace(a,b,1001)
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,step(t,c))
    ax[0].set(xlabel="t",ylabel="Signal Input")
    plt.show()

def solve_ODE(t,C,params):
    pass

def ODE_RHS(t,y,C,params):
    """
    Right hand side of the ODEs for the system.
    S is the signal input, y is the state of the system
    """
    # unpack the parameters
    K = params["K"]
    n = params["n"]
    ta = params["ta"]
    te = params["te"]
    # compute the right hand side
    dy = np.zeros(y.shape)
    dy[0:-1] = (C(t)-y[0:-1])/ta
    dy[-1] = (np.sum(C(t))-np.sum(y))/te
    return dy

def one_signal_ODE():
    params = {"K":1,"n":1,"ta":np.array([4]),"te":0.0001}
    def C_step(t):
        return np.array([-step(t,2)])
    def C_hill(t):
        return np.array([hill(t,1,11)])
    t_span = (0,20)
    t = np.linspace(t_span[0],t_span[1],1001)
    result = scipy.integrate.solve_ivp(ODE_RHS,t_span,np.array([0,0]),args=(C_step,params,),t_eval=t)
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,C_step(t).T)
    ax[0].set(xlabel="t",ylabel="Signal Input")
    ax[0].legend(["C1","C2"])
    ax[1].plot(result.t,result.y.T)
    ax[1].set(xlabel="t",ylabel="States")
    ax[1].legend(["y1","y2","y3"])
    fig.suptitle("One Signal With Hill Function")
    plt.show()

def one_signal_step_vs_hill():
    params = {"K":1,"n":1,"ta":np.array([1]),"te":0.0001}
    def C_step(t):
        return np.array([step(t,1)])
    def C_hill(t):
        return np.array([hill(t,1,11)])
    t_span = (0,5)
    t = np.linspace(t_span[0],t_span[1],1001)
    result_hill = scipy.integrate.solve_ivp(ODE_RHS,t_span,np.array([0,0]),args=(C_hill,params,),t_eval=t)
    fig,ax = plt.subplots(2,2)
    ax[0,0].plot(t,C_hill(t).T)
    ax[0,0].set(xlabel="t",ylabel="Signal Input")
    ax[0,0].legend(["C1","C2"])
    ax[1,0].plot(result_hill.t,result_hill.y.T)
    ax[1,0].set(xlabel="t",ylabel="States")
    ax[1,0].legend(["y1","y2","y3"])
    fig.suptitle("One Signal With Hill Function")
    plt.show()

def one_signal_hill():
    params = {"K":1,"n":1,"ta":np.array([1]),"te":0.0001}
    def C_hill(t):
        return np.array([hill(t,0.5,3)])
    t_span = (0,10)
    t = np.linspace(t_span[0],t_span[1],1001)
    result_hill = scipy.integrate.solve_ivp(ODE_RHS,t_span,np.array([0,0]),args=(C_hill,params,),t_eval=t)
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,C_hill(t).T)
    ax[0].set(xlabel="t",ylabel="Signal Input")
    ax[0].legend(["C1","C2"])
    ax[1].plot(result_hill.t,result_hill.y.T)
    ax[1].set(xlabel="t",ylabel="States")
    ax[1].legend(["y1","y2","y3"])
    fig.suptitle("One Signal With Hill Function")
    plt.savefig("media/one_signal_hill.png")
    # plt.show()

def two_signal_ODE():
    params = {"K":1,"n":1,"ta":np.array([1,5]),"te":0.001}
    def C_step(t):
        return np.array([step(t,2),-1*step(t,2)])
    # def C_hill(t):
    #     return np.array([hill(t,1,10),hill(t,1,10)])
    t_span = (0,20)
    t = np.linspace(t_span[0],t_span[1],1001)
    result = scipy.integrate.solve_ivp(ODE_RHS,t_span,np.array([0,0,0]),args=(C_step,params,),t_eval=t)
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,C_step(t).T)
    ax[0].set(xlabel="t",ylabel="Signal Input")
    ax[0].legend(["C1","C2"])
    ax[1].plot(result.t,result.y.T)
    ax[1].set(xlabel="t",ylabel="States")
    ax[1].legend(["y1","y2","y3"],loc="upper right")
    fig.suptitle("Two Signal ODE")
    plt.savefig("media/two_signal_ODE.png")

# one_signal_hill()
# two_signal_ODE()
# plot_hill()
one_signal_ODE()
