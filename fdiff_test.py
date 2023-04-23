from skfdiff import Model, Simulation
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil
from pyramid import offset_pyramid

def fdiff_animate(U):
    fig,ax = plt.subplots()
    ax.imshow(U[0],cmap='coolwarm')
    ax.set(xlabel='x', ylabel='y')
    def update(i):
        ax.clear()
        ax.imshow(U[i],cmap='coolwarm')
        ax.set(xlabel='x', ylabel='y')
    anim = animation.FuncAnimation(fig, update, frames=U.shape[0], interval=100)
    return anim

def fdiff_animate_3d(x,y,U,stride=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(x,y,indexing="ij")
    min,max = np.min(U),15
    # min,max = np.min(U),np.max(U)
    # ax.plot_surface(X,Y,U[0],cmap='coolwarm',vmin=min,vmax=max)
    ax.plot_surface(X,Y,U[0],cmap='coolwarm')
    ax.set(xlabel='x', ylabel='y')
    # ax.set(zlim=(min,max))
    def update(j):
        i = j*stride
        ax.clear()
        # ax.set(zlim=(min,max))
        # ax.plot_surface(X,Y,U[i],cmap='coolwarm',vmin=min,vmax=max)
        ax.plot_surface(X,Y,U[i],cmap='coolwarm')
        ax.set(xlabel='x', ylabel='y')
    anim = animation.FuncAnimation(fig, update, frames=U.shape[0]//stride, interval=100)
    return anim

def plot_mass(t,U,x,y):
    dx = (x[-1]-x[0])/x.shape[0]
    dy = (y[1]-y[0])/y.shape[0]
    mass = np.sum(U,axis=(1,2))*dx*dy
    fig,ax = plt.subplots()
    ax.plot(t,mass)
    return ax 

def normalise(U,x,y):
    dx = (x[-1]-x[0])/x.shape[0]
    dy = (y[1]-y[0])/y.shape[0]
    return U/(dx*dy*np.sum(U,axis=(1,2))[:,None,None])
    # return U/(dx*dy*np.sum(U[0]))

def save(container,folder_name,stride=1):
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        if input("Folder already exists, overwrite? (y/n)") == "y":
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        else:
            raise FileExistsError("Folder already exists")
    t = container.data.t[::stride]
    x = container.data.x[::stride]
    y = container.data.y[::stride]
    U = container.data.U[::stride]
    S = container.data.S[::stride]
    np.savez(f"{folder_name}/data.npz",t=t,x=x,y=y,U=U,S=S)

def saveframe(i,x,y,U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(x,y,indexing="ij")
    ax.plot_surface(X,Y,U[i],cmap='coolwarm')
    ax.set(xlabel='x', ylabel='y')
    plt.savefig("media/delta_pyramid_fdiff_{}.png".format(i))
    plt.close()

def load(folder_name):
    data = np.load(f"{folder_name}/data.npz")
    t = data["t"]
    x = data["x"]
    y = data["y"]
    U = data["U"]
    S = data["S"]
    return t,x,y,U,S

def pyramid(x, y): 
    X,Y = 2*x-1, 2*y-1
    pyramid = np.where(np.logical_and(Y<=X,Y<=-X),1+Y,X)
    pyramid = np.where(np.logical_and(Y<=X,Y>=-X),1-X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y<=-X),1+X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y>=-X),1-Y,pyramid)
    return 5*pyramid

gaussian = lambda x,y : np.exp(-((x-10)**2+(y-3)**2)/0.05)
parabola = lambda x,y : 1-((x-0.5)**2+(y-0.5)**2)
uniform = lambda x,y : 1


xmax,ymax = 20,10
# nx,ny = 100,50
nx,ny = 40,20
x = np.linspace(0,xmax,nx)
y = np.linspace(0,ymax,ny)
dx = xmax/nx
dy = ymax/ny
delta = lambda x,y : np.where(np.logical_and(x>10-dx,x<10+dx),np.where(np.logical_and(y>3-dy,y<3+dy),1,0),0)
X,Y = np.meshgrid(x,y,indexing="ij")
integral_gaussian = np.sum(gaussian(X,Y))*dx*dy
U  = gaussian(X,Y)/integral_gaussian
# integral_delta = np.sum(delta(X,Y))*dx*dy
# U  = 1e6*delta(X,Y)/integral_delta
print(np.sum(U)*dx*dy)
S = 8*offset_pyramid(X,Y,c=[12,6])
def plot_initial(U,x,y):
    plt.pcolor(X,Y,U)
    plt.vlines(x,0,10)
    plt.hlines(y,0,20)
    plt.axis("equal")
    plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(X,Y,U,cmap="coolwarm")


# Define the model
def run_fig_6_1():
    noflux = {("U","x"):("D*dxU-chi*U*dxS","D*dxU-chi*U*dxS"),("U","y"):("D*dyU-chi*U*dyS","D*dyU-chi*U*dyS")}
    neumann_0 = {("U","x"):("dxU","dxU"),("U","y"):("dyU","dyU")}
    neumann_0016 = {("U","x"):("dxU-0.016","dxU-0.016"),("U","y"):("dyU-0.016","dyU-0.016")}
    model = Model(["D*(dxxU+dyyU)-chi*(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
                  ["U(x,y)","S(x,y)"],parameters = ["D","chi","k"],boundary_conditions=noflux)
    # model = Model(["D*(dxxU+dyyU)-(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
    #               ["U(x,y)","S(x,y)"],parameters = ["k","D"],boundary_conditions=noflux)
    x = np.linspace(0,20,80)
    y = np.linspace(0,10,40)
    X,Y = np.meshgrid(x,y,indexing="ij")
    U  = delta(X,Y)
    S = 8*offset_pyramid(X,Y,c=[12,6])
    # U = uniform(X,Y)s
    # S = gaussian(X,Y)
    k = 0
    b = 1 
    te = 0 
    ta = 1
    lambda0 = 1
    s = 0.1 
    D = (s**2)/(2*lambda0)
    chi = (b*s**2*ta)/(2*lambda0*(1+te*lambda0)*(1+ta*lambda0))
    initial_fields = model.Fields(x=x,y=y,U=U,S=S,k=k,D=D,chi=chi)
    simulation = model.init_simulation(initial_fields,dt=0.1,tmax=2000,scheme="Theta",theta=0.5)
    # simulation = Simulation(model,initial_fields,dt=0.001,tmax=0.5)
    container = simulation.attach_container()
    tmax,last_fields = simulation.run()
    folder_name = f"data/test_fdiff_k_{k}_full_final"
    save(container,folder_name,stride=100)

def plot_6_1_initial():
    pass

def plot_sin_gaussian_initial():
    U  = np.ones(X.shape)
    m, n = 5,5
    S1 = np.cos(m*np.pi*X/10)*np.cos(n*np.pi*Y/10)
    S2 = np.exp(-((X-5)**2+(Y-5)**2)/2)
    fig = plt.figure()
    ax0 = fig.add_subplot(121,projection="3d")
    ax0.plot_surface(X,Y,U,cmap="coolwarm")
    ax1 = fig.add_subplot(122,projection="3d")
    ax1.plot_surface(X,Y,S1,cmap="coolwarm")
    plt.show()

def run_sin_gaussian():
    noflux = {("U","x"):("D*dxU-chi*U*dxS","D*dxU-chi*U*dxS"),("U","y"):("D*dyU-chi*U*dyS","D*dyU-chi*U*dyS")}
    model = Model(["D*(dxxU+dyyU)-chi*(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
                  ["U(x,y)","S(x,y)"],parameters = ["D","chi","k"],boundary_conditions=noflux)
    # model = Model(["D*(dxxU+dyyU)-(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
    #               ["U(x,y)","S(x,y)"],parameters = ["k","D"],boundary_conditions=noflux)
    x = np.linspace(0,10, 80)
    y = np.linspace(0,10, 80)
    X,Y = np.meshgrid(x,y,indexing="ij")
    # uniformly distributed cells with a gaussian signal and oscillating signal
    U  = np.ones(X.shape)
    m, n = 5,5
    S1 = np.cos(m*np.pi*X/10)*np.cos(n*np.pi*Y/10)
    S2 = np.exp(-((X-5)**2+(Y-5)**2)/2)
    k = 0
    b = 1 
    te = 0 
    ta = 1
    lambda0 = 1
    s = 0.1 
    D = (s**2)/(2*lambda0)
    chi = (b*s**2*ta)/(2*lambda0*(1+te*lambda0)*(1+ta*lambda0))
    initial_fields = model.Fields(x=x,y=y,U=U,S=S,k=k,D=D,chi=chi)
    simulation = model.init_simulation(initial_fields,dt=0.1,tmax=2000,scheme="Theta",theta=0.5)
    # simulation = Simulation(model,initial_fields,dt=0.001,tmax=0.5)
    container = simulation.attach_container()
    tmax,last_fields = simulation.run()
    folder_name = f"data/test_fdiff_k_{k}_full_final"
    save(container,folder_name,stride=100)

if __name__ == "__main__":
    run_fig_6_1()
    # k = 0
    folder_name = f"data/test_fdiff_k_0_full_final"
    t,x,y,U,S = load(folder_name)
    U = U
    S = S
    # # for some reason x and y are just 0 
    x = np.linspace(0,20,80)
    y = np.linspace(0,10,40)
    X,Y = np.meshgrid(x,y,indexing="ij")
    t = np.linspace(0,2000,201)
    U = normalise(U,x,y)
    # plot_mass(t,U,x,y)
    # plt.show()
    # to make sum to one as histogram does 
    nx,ny = x.shape,y.shape
    anim = fdiff_animate_3d(x,y,U*10,stride=1)
    plt.show()

    # anim.save("media/delta_pyramid_fdiff.mp4")

    # saveframe(20,x,y,U)
    # saveframe(-1,x,y,U)

    # plt.show()