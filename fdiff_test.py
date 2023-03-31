from skfdiff import Model, Simulation
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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

def fdiff_animate_3d(x,y,U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,U[0],cmap='coolwarm',vmin=np.min(U),vmax=np.max(U))
    ax.set(xlabel='x', ylabel='y')
    min,max = np.min(U),np.max(U)
    ax.set(zlim=(min,max))
    def update(i):
        ax.clear()
        ax.set(zlim=(min,max))
        ax.plot_surface(X,Y,U[i],cmap='coolwarm',vmin=min,vmax=max)
        ax.set(xlabel='x', ylabel='y')
    anim = animation.FuncAnimation(fig, update, frames=U.shape[0], interval=100)
    return anim
def plot_mass(t,U):
    mass = np.sum(U,axis=(1,2))
    fig,ax = plt.subplots()
    ax.plot(t,mass)
    return ax 
def normalise(U):
    return U/np.sum(U,axis=(1,2))
def save(container,folder_name):
    os.mkdir(folder_name)
    t = container.data.t
    x = container.data.x
    y = container.data.y
    U = container.data.U
    S = container.data.S
    np.savez(f"{folder_name}/data.npz",t=t,x=x,y=y,U=U,S=S)

def load(folder_name):
    data = np.load(f"{folder_name}/data.npz")
    t = data["t"]
    x = data["x"]
    y = data["y"]
    U = data["U"]
    S = data["S"]
    return t,x,y,U,S

delta = lambda x,y : np.where(np.logical_and(x>0.6,x<0.8),np.where(np.logical_and(y>0.6,y<0.8),1,0),0)
def pyramid(x, y): 
    X,Y = 2*x-1, 2*y-1
    pyramid = np.where(np.logical_and(Y<=X,Y<=-X),1+Y,X)
    pyramid = np.where(np.logical_and(Y<=X,Y>=-X),1-X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y<=-X),1+X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y>=-X),1-Y,pyramid)
    return 5*pyramid
        
# Define the model
# "dxxU+dyyU-(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)"d
model = Model(["dxxU+dyyU-(dxU*dxS+dyU*dyS+U*dxxS+U*dyyS)","k*(dxxS+dyyS)"],
              ["U(x,y)","S(x,y)"],parameters = "k",boundary_conditions="noflux")
x = np.linspace(0,1,50)
y = np.linspace(0,1,50)
X,Y = np.meshgrid(x,y)
U  = delta(X,Y)
S = pyramid(X,Y)
initial_fields = model.Fields(x=x,y=y,U=U,S=S,k=1)
simulation = Simulation(model,initial_fields,dt=0.001,tmax=0.2)
container = simulation.attach_container()
tmax,last_fields = simulation.run()
folder_name = "data/test_fdiff"
save(container,folder_name)

folder_name = "data/test_fdiff"
t,x,y,U,S = load(folder_name)
# U = normalise(U)
plot_mass(t,S)
plt.show()
anim = fdiff_animate_3d(x,y,S)
plt.show()
anim = fdiff_animate_3d(x,y,U)
plt.show()
# anim.save("media/delta_pyramid_fdiff.mp4")

# def saveframe(i,container):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X,Y = np.meshgrid(container.data.x,container.data.y)
#     ax.plot_surface(X,Y,container.data.U[i],cmap='coolwarm')
#     ax.set(xlabel='x', ylabel='y')
#     plt.savefig("media/delta_pyramid_fdiff_{}.png".format(i))
#     plt.close()

# saveframe(10,container)
# saveframe(-1,container)

# plt.show()