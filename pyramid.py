import numpy as np 
import matplotlib.pyplot as plt 

def pyramid(x, y): 
    X,Y = 2*x-1, 2*y-1
    pyramid = np.where(np.logical_and(Y<=X,Y<=-X),1+Y,X)
    pyramid = np.where(np.logical_and(Y<=X,Y>=-X),1-X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y<=-X),1+X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y>=-X),1-Y,pyramid)
    return pyramid

def grad_pyramid(x, y):
    X,Y = 2*x-1, 2*y-1
    X_component = np.where(np.logical_and(Y<=X,Y<=-X),2,Y)
    X_component = np.where(np.logical_and(Y<=X,Y>=-X),-2,X_component)
    X_component = np.where(np.logical_and(Y>=X,Y<=-X),2,X_component)
    X_component = np.where(np.logical_and(Y>=X,Y>=-X),-2,X_component)
    Y_component = np.where(np.logical_and(Y<=X,Y<=-X),2,Y)
    Y_component = np.where(np.logical_and(Y<=X,Y>=-X),-2,Y_component)
    Y_component = np.where(np.logical_and(Y>=X,Y<=-X),2,Y_component)
    Y_component = np.where(np.logical_and(Y>=X,Y>=-X),-2,Y_component)
    return [X_component, Y_component]

def lap_pyramid(x, y):
    return np.zeros(x.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0,1, 1000)
y = np.linspace(0,1, 1000)
X, Y = np.meshgrid(x, y)
Z = pyramid(X, Y)
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set(xlabel='x', ylabel='y', zlabel='z')
plt.show()