import numpy as np
import matplotlib.pyplot as plt


def pyramid(x, y):
    ax, bx = x.min(), x.max()
    ay, by = y.min(), y.max()
    X, Y = 2 * (x - ax) / (bx - ax) - 1, 2 * (y - ay) / (by - ay) - 1
    # testing if go to right side of pyramid
    # X,Y = -X,-Y
    pyramid = np.where(np.logical_and(Y <= X, Y <= -X), 1 + Y, X)
    pyramid = np.where(np.logical_and(Y <= X, Y >= -X), 1 - X, pyramid)
    pyramid = np.where(np.logical_and(Y >= X, Y <= -X), 1 + X, pyramid)
    pyramid = np.where(np.logical_and(Y >= X, Y >= -X), 1 - Y, pyramid)
    return 1 * pyramid


def offset_pyramid(x, y, c):
    ax, bx = x.min(), x.max()
    ay, by = y.min(), y.max()
    cx,cy = c
    X, Y = 2 * (x - ax) / (bx - ax) - 1, 2 * (y - ay) / (by - ay) - 1
    CX, CY = 2 * (cx - ax) / (bx - ax) - 1, 2 * (cy - ay) / (by - ay) - 1
    # z gradients of the pyramids
    zkxp, zkyp = 1 / (CX + 1), 1 / (CY + 1)
    zkxm, zkym = 1 / (1 - CX), 1 / (1 - CY)
    # inequality gradients 
    kpp = (1-CY)/(1-CX)
    kpm = (-1-CY)/(1-CX)
    kmp = (1-CY)/(-1-CX)
    kmm = (-1-CY)/(-1-CX)
    # testing if go to right side of pyramid
    # X,Y = -X,-Y   
    pyramid = np.where(np.logical_and((Y-CY)<= kmm*(X-CX), (Y-CY)<= kpm*(X-CX)), zkyp*(1 + Y), X) #bottom
    pyramid = np.where(np.logical_and((Y-CY)<= kpp*(X-CX), (Y-CY)>= kpm*(X-CX)), zkxm*(1 - X), pyramid) #right
    pyramid = np.where(np.logical_and((Y-CY)>= kmm*(X-CX), (Y-CY)<= kmp*(X-CX)), zkxp*(1 + X), pyramid) #left
    pyramid = np.where(np.logical_and((Y-CY)>= kpp*(X-CX), (Y-CY)>= kmp*(X-CX)), zkym*(1 - Y), pyramid) #top
    # pyramid = np.where(np.logical_and((Y-CY)<= kmm*(X-CX), (Y-CY)<= kpm*(X-CX)), np.ones(X.shape), np.zeros(X.shape)) #bottom
    # pyramid = np.where(np.logical_and((Y-CY)<= kpp*(X-CX), (Y-CY)>= kpm*(X-CX)), 2*np.ones(X.shape), pyramid) #right
    # pyramid = np.where(np.logical_and((Y-CY)>= kmm*(X-CX), (Y-CY)<= kmp*(X-CX)), 3*np.ones(X.shape), pyramid) #left
    # pyramid = np.where(np.logical_and((Y-CY)>= kpp*(X-CX), (Y-CY)>= kmp*(X-CX)), 4*np.ones(X.shape), pyramid) #top
    return 1 * pyramid


def grad_pyramid(x, y):
    X, Y = 2 * x - 1, 2 * y - 1
    X_component = np.where(np.logical_and(Y <= X, Y <= -X), 2, Y)
    X_component = np.where(np.logical_and(Y <= X, Y >= -X), -2, X_component)
    X_component = np.where(np.logical_and(Y >= X, Y <= -X), 2, X_component)
    X_component = np.where(np.logical_and(Y >= X, Y >= -X), -2, X_component)
    Y_component = np.where(np.logical_and(Y <= X, Y <= -X), 2, Y)
    Y_component = np.where(np.logical_and(Y <= X, Y >= -X), -2, Y_component)
    Y_component = np.where(np.logical_and(Y >= X, Y <= -X), 2, Y_component)
    Y_component = np.where(np.logical_and(Y >= X, Y >= -X), -2, Y_component)
    return [X_component, Y_component]


def lap_pyramid(x, y):
    return np.zeros(x.shape)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = np.linspace(0, 20, 1000)
    y = np.linspace(0, 10, 1000)
    c = [12,6]
    X, Y = np.meshgrid(x, y)
    Z = offset_pyramid(X, Y, c)
    ax.plot_surface(X, Y, Z, cmap="coolwarm")
    # ax.set(xlim=(0, 2), ylim=(0, 2))
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    plt.show()
