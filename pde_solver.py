import numpy as np
import numpy.linalg as lag
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
        self.mu = self.dt / self.dx**2
        self.eta = self.dt / (2 * self.dx)
        self.n_space_points = int(self.L / self.dx)
        self.U = np.ones([self.M, self.n_space_points, self.n_space_points])
        self.X, self.Y = np.meshgrid(
            np.linspace(0, 1, self.n_space_points),
            np.linspace(0, 1, self.n_space_points),
        )
        self.U[0, :, :] = self.initial_condition(self.X, self.Y)
        self.chemical_values = self.chemical(self.X, self.Y)
        self.chemical_gradient = [component/self.dx for component in np.gradient(self.chemical_values)]
        for m in range(self.M - 1):
            if self.verbose and m % 100 == 0:
                print("Time step: ", m)
            U_right = self.U[m, 2:, 1:-1]
            U_left = self.U[m, :-2, 1:-1]
            U_up = self.U[m, 1:-1, 2:]
            U_down = self.U[m, 1:-1, :-2]
            U_center = self.U[m, 1:-1, 1:-1]
            gradient_right = self.chemical_gradient[0][1:-1, 2:]
            gradient_left = self.chemical_gradient[0][1:-1, :-2]
            gradient_up = self.chemical_gradient[1][2:, 1:-1]
            gradient_down = self.chemical_gradient[1][:-2, 1:-1]
            self.U[m + 1, 1:-1, 1:-1] = (
                U_center
                + self.mu * (U_right + U_left + U_up + U_down - 4 * U_center)
                - self.eta* (
                    gradient_right * U_right
                    - gradient_left * U_left
                    + gradient_up * U_up
                    - gradient_down * U_down
                )
            )
            # setting no flux boundary conditions
            # TODO this is wrong! It should invlove the gradient of the chemical
            self.U[m + 1, 0, :] = self.U[m + 1, 1, :]
            self.U[m + 1, -1, :] = self.U[m + 1, -2, :]
            self.U[m + 1, :, 0] = self.U[m + 1, :, 1]
            self.U[m + 1, :, -1] = self.U[m + 1, :, -2]
            # self.chemical = self.chemical + self.dt*self.chemical_ODE(self.chemical,self.X,self.Y)
            # self.chemical_gradient = np.gradient(self.chemical)

    def animate(self):
        """Animates the solution."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.imshow(self.U_data[0, :, :], cmap="coolwarm")

        def animate(i):
            ax.imshow(self.U_data[i, :, :], cmap="coolwarm")

        anim = animation.FuncAnimation(fig, animate, frames=self.M, interval=1)
        return fig, ax, anim

    def animate_3d(self,stride =1):
        """Animates the solution."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        min = np.min(self.U)
        max = np.max(self.U)
        ax.set_zlim(min, max)
        surface = ax.plot_surface(self.X, self.Y, self.U[0, :, :], cmap="coolwarm",vmin=min, vmax=max)

        def animate(j):
            i = j*stride
            ax.clear()
            ax.set_title("Time step: " + str(i))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(min, max)
            surface = ax.plot_surface(self.X, self.Y, self.U[i, :, :], cmap="coolwarm",vmin=min, vmax=max)
        anim = animation.FuncAnimation(fig, animate, frames=self.M//stride, interval=1)
        return fig, ax, anim
