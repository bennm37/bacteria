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
    X,Y = np.meshgrid(x,y)
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
def plot_mass(t,U):
    mass = np.sum(U,axis=(1,2))
    fig,ax = plt.subplots()
    ax.plot(t,mass)
    return ax 
def normalise(U):
    return U/np.sum(U,axis=(1,2))[:,np.newaxis,np.newaxis]
def save(container,folder_name,stride=1):
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        if input("Folder already exists, overwrite? (y/n)") == "y":
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        else:
            raise FileExistsError("Folder already exists")
    t = container.data.t
    x = container.data.x
    y = container.data.y
    U = container.data.U
    S = container.data.S
    np.savez(f"{folder_name}/data.npz",t=t[::stride],x=x[::stride],y=y[::stride],U=U[::stride],S=S[::stride])

def saveframe(i,x,y,U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(x,y)
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

delta = lambda x,y : 16000*np.where(np.logical_and(x>10,x<10.8),np.where(np.logical_and(y>3,y<3.4),1,0),0)/(0.4*0.8)
parabola = lambda x,y : 1-((x-0.5)**2+(y-0.5)**2)
uniform = lambda x,y : 1
gaussian = lambda x,y : 0.1*np.exp(-((x-0.5)**2+(y-0.5)**2)/0.05)

# x = np.linspace(0,20,50)
# y = np.linspace(0,10,50)
# X,Y = np.meshgrid(x,y)
# U  = delta(X,Y)
# S = 8*offset_pyramid(X,Y,c=[12,6])
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(X,Y,S)
# plt.show()

# Define the model
def run():
    noflux = {("U","x"):("D*dxU-chi*U*dxS","D*dxU-chi*U*dxS"),("U","y"):("D*dyU-chi*U*dyS","D*dyU-chi*U*dyS")}
    neumann_0 = {("U","x"):("dxU","dxU"),("U","y"):("dyU","dyU")}
    neumann_0016 = {("U","x"):("dxU-0.016","dxU-0.016"),("U","y"):("dyU-0.016","dyU-0.016")}
    model = Model(["D*(dxxU+dyyU)-chi*(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
                  ["U(x,y)","S(x,y)"],parameters = ["D","chi","k"],boundary_conditions=noflux)
    # model = Model(["D*(dxxU+dyyU)-(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
    #               ["U(x,y)","S(x,y)"],parameters = ["k","D"],boundary_conditions=noflux)
    x = np.linspace(0,20,50)
    y = np.linspace(0,10,50)
    X,Y = np.meshgrid(x,y)
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
    simulation = model.init_simulation(initial_fields,dt=0.1,tmax=20,scheme="Theta",theta=0.5)
    # simulation = Simulation(model,initial_fields,dt=0.001,tmax=0.5)
    container = simulation.attach_container()
    tmax,last_fields = simulation.run()
    folder_name = f"data/test_fdiff_k_{k}_full_test"
    save(container,folder_name,stride=1)

# run()
k = 0
folder_name = f"data/test_fdiff_k_{k}_full_32000"
t,x,y,U,S = load(folder_name)
U = normalise(U)*(50**2)
plot_mass(t,U)
plt.show()
anim = fdiff_animate_3d(x,y,U,stride=100)
plt.show()
# anim.save("media/delta_pyramid_fdiff.mp4")

# saveframe(20,x,y,U)
# saveframe(-1,x,y,U)

# plt.show()