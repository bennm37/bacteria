import numpy as np
import numpy.linalg as lag
import scipy.linalg as slag
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PDE_Solver:
    def __init__(self, parameters, verbose=False):
        """Solves the parabolic PDE for the chemical and cell concentrations."""
        self.parameters = parameters
        self.verbose = verbose
        self.chemical = parameters["chemical"]
        self.taxis_strength = parameters["taxis_strength"]
        self.initial_condition = parameters["initial_condition"]
        self.N = parameters["N"]
        self.L = parameters["L"]
        self.M = parameters["M"]
        self.lambda0 = parameters["lambda0"]
        self.dt = parameters["dt"]
        self.dx = parameters["dx"]
        self.te = parameters["te"]
        self.ta = parameters["ta"]
        self.dimension = parameters["dimension"]

    def initialise(self):
        """Initialises the grid and the chemical and cell concentrations."""
        pass

    def rescale(self):
        """Rescales the chemical and cell concentrations."""
        pass

    def explicit_solve(self):
        """Solves the PDE."""
        self.dx = self.dx / self.L
        self.X, self.Y = np.meshgrid(
            np.arange(0, 1, self.dx),
            np.arange(0, 1, self.dx),
        )
        self.n_space_points = self.X.shape[0]
        self.mu = self.dt / self.dx**2
        self.eta = self.dt / (2 * self.dx)
        self.U = np.ones([self.M, self.n_space_points, self.n_space_points])
        # setting up mesh with ghost points
        self.U[0, :, :] = self.initial_condition(self.X, self.Y)
        self.S = self.chemical(self.X, self.Y)
        self.U[0, 0, :] = self.U[0, 1, :] / (1 + self.S[1, :] - self.S[0, :])  # left
        self.U[0, -1, :] = self.U[0, -2, :] / (1 - (self.S[-1, :] - self.S[-2, :])) # right
        self.U[0, :, 0] = self.U[0, :, 1] / (1 + self.S[:, 1] - self.S[:, 0]) # bottom
        self.U[0, :, -1] = self.U[0, :, -2] / (1 - (self.S[:, -1] - self.S[:, -2])) # top
        for m in range(self.M - 1):
            if self.verbose and m % 100 == 0:
                print("Time step: ", m)
            U_right = self.U[m, 2:, 1:-1]
            U_left = self.U[m, :-2, 1:-1]
            U_up = self.U[m, 1:-1, 2:]
            U_down = self.U[m, 1:-1, :-2]
            U_center = self.U[m, 1:-1, 1:-1]
            S_left = self.S[:-2, 1:-1]
            S_right = self.S[2:, 1:-1]
            S_up = self.S[1:-1, 2:]
            S_down = self.S[1:-1, :-2]
            S_center = self.S[1:-1, 1:-1]
            self.U[m + 1, 1:-1, 1:-1] = (
                U_center
                + self.mu * (U_right + U_left + U_up + U_down - 4 * U_center)
                - self.mu * (S_right + S_left + S_up + S_down - 4 * S_center)
                - self.mu / 4 * (U_right - U_left) * (S_right - S_left)
                - self.mu / 4 * (U_up - U_down) * (S_up - S_down)
            )
            # setting 
            self.U[m + 1, 0, :] = self.U[m + 1, 1, :] / (1 + self.S[1, :] - self.S[0, :])  # left
            self.U[m + 1, -1, :] = self.U[m + 1, -2, :] / (1 - (self.S[-1, :] - self.S[-2, :])) # right
            self.U[m + 1, :, 0] = self.U[m + 1, :, 1] / (1 + self.S[:, 1] - self.S[:, 0]) # bottom
            self.U[m + 1, :, -1] = self.U[m + 1, :, -2] / (1 - (self.S[:, -1] - self.S[:, -2])) # top
            # checking fluxes for debugging 
            self.flux_left = (self.U[m+1,1,:]-self.U[m+1,0,:])/(self.dx)-self.U[m+1,0,:]*(self.S[1,:]-self.S[0,:])/(self.dx)
            self.flux_right = (self.U[m+1,-1,:]-self.U[m+1,-2,:])/(self.dx)-self.U[m+1,0,:]*(self.S[-1,:]-self.S[-2,:])/(self.dx)
            self.flux_down = (self.U[m+1,:,1]-self.U[m+1,:,0])/(self.dx)-self.U[m+1,:,0]*(self.S[:,1]-self.S[:,0])/(self.dx)
            self.flux_up = (self.U[m+1,:,-1]-self.U[m+1,:,-2])/(self.dx)-self.U[m+1,:,-1]*(self.S[:,-1]-self.S[:,-2])/(self.dx)
            # self.chemical = self.chemical + self.dt*self.chemical_ODE(self.chemical,self.X,self.Y)
            # self.chemical_gradient = np.gradient(self.chemical)

    def explicit_solve2(self,gradient,laplacian):
        """Solves the PDE using an explicit scheme."""
        self.dx = self.dx / self.L
        self.X, self.Y = np.meshgrid(
            np.arange(0, 1, self.dx),
            np.arange(0, 1, self.dx),
        )
        self.n_space_points = self.X.shape[0]
        self.mu = self.dt / self.dx**2
        self.eta = self.dt / (2 * self.dx)
        self.U = np.ones([self.M, self.n_space_points, self.n_space_points])
        # setting up mesh with ghost points
        self.U[0, :, :] = self.initial_condition(self.X, self.Y)
        self.S = self.chemical(self.X, self.Y)
        self.U[0, 0, :] = self.U[0, 1, :] / (1 + self.S[1, :] - self.S[0, :])

    def animate(self):
        """Animates the solution."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.imshow(self.U_data[0, :, :], cmap="coolwarm")

        def update(i):
            ax.imshow(self.U_data[i, :, :], cmap="coolwarm")

        anim = animation.FuncAnimation(fig, update, frames=self.M, interval=1)
        return fig, ax, anim

    def implicit_solve(self):
        """Solves the PDE using an implicit scheme."""
        if self.dimension ==1:
            self.dx = self.dx/self.L
            self.x = np.arange(0, 1+self.dx, self.dx) #1+dx to include endpoint 
            self.n_space_points = len(self.x)-1
            # construct the matrix
            self.S = self.chemical(self.x)
            # compute interior S gradient points using central difference
            self.S1 = (self.S[2:]-self.S[0:-2])
            self.S2 = (self.S[2:]+self.S[0:-2]-2*self.S[1:-1])
            self.A = np.zeros((self.n_space_points, self.n_space_points))
            self.A[1:-1,0:-2] += np.diag(1+self.S1[:-1]/4, 0)/self.dx**2
            self.A[1:-1,2:] += np.diag(1-self.S1[:-1]/4, 0)/self.dx**2
            # self.A += np.diag(self.S1[1:]/4+1, -1) 
            self.A[1:-1,1:-1] += np.diag(-2*np.ones(self.n_space_points-2),0)/self.dx**2
            plt.show()



    def animate_3d(self, stride=1):
        """Animates the solution."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        min = np.min(self.U)
        max = np.max(self.U)
        ax.set_zlim(min, max)
        surface = ax.plot_surface(
            self.X, self.Y, self.U[0, :, :], cmap="coolwarm", vmin=min, vmax=max
        )

        def update(j):
            i = j * stride
            if i%10 == 0:
                print("Animating for timestep: ", i)
            ax.clear()
            ax.set_title("Time step: " + str(i))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(min, max)
            surface = ax.plot_surface(
                self.X, self.Y, self.U[i, :, :], cmap="coolwarm", vmin=min, vmax=max
            )

        anim = animation.FuncAnimation(
            fig, update, frames=self.M // stride, interval=1
        )
        return fig, ax, anim
