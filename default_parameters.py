import numpy as np
sins = lambda x,y: np.sin(x-5)*np.sin(y-5)
cone = lambda x,y: -np.sqrt((x-5)**2+(y-5)**2)
# parabola = lambda x,y: -(x-5)**2-(y-5)**2
def parabola(x,y):
    arr = -(x-5)**2-(y-5)**2
    return arr
def pyramid(x, y): 
    ax,bx = x.min(),x.max()
    ay,by = y.min(),y.max()
    X,Y = 2*(x-ax)/(bx-ax)-1, 2*(y-ay)/(by-ay)-1
    # testing if go to right side of pyramid
    # X,Y = -X,-Y
    pyramid = np.where(np.logical_and(Y<=X,Y<=-X),1+Y,X)
    pyramid = np.where(np.logical_and(Y<=X,Y>=-X),1-X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y<=-X),1+X,pyramid)
    pyramid = np.where(np.logical_and(Y>=X,Y>=-X),1-Y,pyramid)
    return 1*pyramid
parameters = {
    "M": 5000,
    "N": 10000,
    "L": 10,
    "te": 0.00001,
    "ta": 1,
    "dt": 0.01,
    "dx": 0.01,
    "chemical": pyramid,
    "taxis_strength": 1,
    "speed": 0.1,
    "n_states":2,
    "state_ODE": lambda C,x,y: np.array([(C-x-y)/parameters["te"],(C-y)/parameters["ta"]]).T,
    "lambda0": 1,
    "initial_condition": "delta",
    "dimension": 2,
    "seed": 3
}
