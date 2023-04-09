import numpy as np
import numpy.linalg as lag
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import os 
import shutil as sh
import glob


class Colony:
    def __init__(self, parameters, verbose=False):
        self.parameters = parameters
        self.verbose = verbose
        self.N = parameters["N"]
        self.Lx = parameters["Lx"]
        self.Ly = parameters["Ly"]
        self.M = parameters["M"]
        self.dt = parameters["dt"]
        self.chemical = parameters["chemical"]
        self.taxis_strength = parameters["taxis_strength"]
        self.speed = parameters["speed"]
        self.state_ODE = parameters["state_ODE"]
        self.initial_condition = parameters["initial_condition"]
        self.dimension = parameters["dimension"]
        self.seed = parameters["seed"]
        self.lambda0 = parameters["lambda0"]
        self.lambdas = np.ones(self.N)*self.lambda0
        self.n_states = parameters["n_states"]
        self.overwrite = parameters["overwrite"]
        self.array = parameters["array"]
        self.save_frequency = parameters["save_frequency"]
        self.foldername = parameters["foldername"]
        self.save = parameters["save"]
        self.turn = 1
        np.random.seed(self.seed)
        self.initialise_cells()

    def initialise_cells(self):
        self.locations = np.zeros([self.N, self.dimension])
        self.velocities = np.zeros([self.N, self.dimension])
        self.states = np.zeros([self.N, self.n_states])
        if self.initial_condition == "uniform":
            self.locations = np.concatenate(
                [np.random.uniform(0,self.Lx,self.N), np.random.uniform(0,self.Ly,self.N)], axis=1
            )
            # self.locations = np.random.uniform(0, self.L, [self.N, self.dimension])
            self.thetas = np.random.uniform(-np.pi, np.pi, [self.N, 1])
            self.speeds = self.speed * np.ones([self.N, 1])
            self.velocities = np.concatenate(
                [np.cos(self.thetas), np.sin(self.thetas)], axis=1
            )
            self.velocities = self.velocities * self.speeds

        elif self.initial_condition == "delta":
            # self.locations = np.random.uniform(0.6*self.L, 0.8*self.L, [self.N, self.dimension])
            self.locations = np.ones([self.N, self.dimension])*[10,3]
            self.thetas = np.random.uniform(-np.pi, np.pi, [self.N, 1])
            self.speeds = self.speed * np.ones([self.N, 1])
            self.velocities = np.concatenate(
                [np.cos(self.thetas), np.sin(self.thetas)], axis=1
            )
            self.velocities = self.velocities * self.speeds

        else:
            raise ValueError("Initial condition not recognised")

    def run(self):
        if self.array:
            self.location_data = np.zeros([self.M//self.save_frequency, self.N, self.dimension])
            # self.state_data = np.zeros([self.M, self.N, self.n_states])
            # self.signal_data = np.zeros([self.M, self.N])
            # self.velocity_data = np.zeros([self.M, self.N, self.dimension])
        if self.save.lower() != "none":
            try:
                os.mkdir(self.foldername)
            except FileExistsError:
                if self.overwrite:
                    sh.rmtree(self.foldername)
                    os.mkdir(self.foldername)
                elif input(f"Delete Folder {self.foldername}?    ").lower() in ["y","yes"]:
                    sh.rmtree(self.foldername)
                    os.mkdir(self.foldername)
                else:
                    raise ValueError("Folder already exists")
                
        for i in range(self.M):
            if self.verbose and i % 100 == 0:
                print(f"Ran up to t = {i*self.dt}")
                # self.plot_cells()
                # plt.show()
            
            t = i * self.dt
            self.locations = (self.locations+self.velocities * self.dt)
            # implementing no-flux boundary conditions
            self.locations[:,0] = np.where(self.locations[:,0]<0,np.zeros_like(self.locations[:,0]),self.locations[:,0])
            self.locations[:,0] = np.where(self.locations[:,0]>self.Lx,self.Lx*np.ones_like(self.locations[:,0]),self.locations[:,0])
            self.locations[:,1] = np.where(self.locations[:,1]<0,np.zeros_like(self.locations[:,1]),self.locations[:,1])
            self.locations[:,1] = np.where(self.locations[:,1]>self.Ly,self.Ly*np.ones_like(self.locations[:,1]),self.locations[:,1])
            self.update_states()
            self.update_velocities()
            if self.array:
                if i%self.save_frequency==0:    
                    j = int(i/self.save_frequency)
                    self.location_data[j] = self.locations
                    # self.state_data[j] = self.states
                    # self.signal_data[j] = self.signal
                    # self.velocity_data[j] = self.velocities
            if self.save == "all":
                if i%self.save_frequency==0:
                    np.savez(f"{self.foldername}/data_{np.round(i*self.dt,2)}",locations=self.locations,states=self.states,velocities=self.velocities,signal=self.signal)
            elif self.save == "locations":
                if i%self.save_frequency==0:
                    np.savez(f"{self.foldername}/data_{np.round(i*self.dt,2)}",locations=self.locations)
            

        if self.array:
            return self.location_data
            # return self.location_data, self.state_data, self.velocity_data, self.signal_data
        else:
            return None
    
    def update_states(self):
        self.signal = self.chemical(self.locations[:,0],self.locations[:,1])
        self.states += self.state_ODE(self.signal,self.states[:,0],self.states[:,1])*self.dt
        # this is the limiting case te = 0 and y1 instantly adjusts to y2
        self.lambdas = self.lambda0-self.taxis_strength*(self.signal-self.states[:,1])
        
    def rotation_kernel(self,theta):
        return np.random.uniform(-np.pi,np.pi,self.N)

    def update_velocities(self):
        r1 = np.random.uniform(0, 1, self.N)
        # vectorising this 10x speedup. Are reshapes necessary/slow?
        self.thetas = np.where(r1<self.lambdas*self.dt,self.rotation_kernel(self.thetas).reshape(self.N),self.thetas.reshape(self.N))
        self.thetas = self.thetas.reshape(self.N,1)

        # This is by far the slowest part of the code
        # for i,theta in enumerate(self.thetas):
        #     if r1[i]<self.lambdas[i]*self.dt:
        #         self.thetas[i] = self.rotation_kernel(theta)
        self.velocities = self.speeds*np.concatenate([np.cos(self.thetas),np.sin(self.thetas)],axis=1)
    
    def update_drug(self):
        pass

    def plot_cells(self):
        fig, ax = plt.subplots()
        speeds = lag.norm(self.velocities, axis=1)
        cmap = cm.get_cmap("coolwarm")
        ax.quiver(
            self.locations[:, 0],
            self.locations[:, 1],
            self.velocities[:, 0],
            self.velocities[:, 1],
            color=cmap(speeds),
            scale=20,
        )
        # ax.quiver(self.locations[:,0],self.locations[:,1],self.velocities[:,0],self.velocities[:,1],cmap="viridis")
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        ax.axis("equal")
        ax.axis("off")
        plt.show()
