import numpy as np
import numpy.linalg as lag
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


class Colony:
    def __init__(self, parameters, verbose=False):
        self.parameters = parameters
        self.verbose = verbose
        self.N = parameters["N"]
        self.L = parameters["L"]
        self.M = parameters["M"]
        self.dt = parameters["dt"]
        self.chemical = parameters["chemical"]
        self.taxis_strength = parameters["taxis_strength"]
        self.state_ODE = parameters["state_ODE"]
        self.initial_condition = parameters["initial_condition"]
        self.dimension = parameters["dimension"]
        self.seed = parameters["seed"]
        self.lambda0 = parameters["lambda0"]
        self.lambdas = np.ones(self.N)*self.lambda0
        self.turn = 1
        self.n_states = 2
        np.random.seed(self.seed)
        self.initialise_cells()

    def initialise_cells(self):
        self.locations = np.zeros([self.N, self.dimension])
        self.velocities = np.zeros([self.N, self.dimension])
        self.states = np.zeros([self.N, self.n_states])
        if self.initial_condition == "uniform":
            self.locations = np.random.uniform(0, self.L, [self.N, self.dimension])
            self.thetas = np.random.uniform(-np.pi, np.pi, [self.N, 1])
            self.speeds = np.random.uniform(1,1, [self.N, 1])
            self.velocities = np.concatenate(
                [np.cos(self.thetas), np.sin(self.thetas)], axis=1
            )
            self.velocities = self.velocities * self.speeds

        else:
            raise ValueError("Initial condition not recognised")

    def run(self):
        self.location_data = np.zeros([self.M, self.N, self.dimension])
        self.state_data = np.zeros([self.M, self.N, self.n_states])
        self.signal_data = np.zeros([self.M, self.N])
        self.velocity_data = np.zeros([self.M, self.N, self.dimension])
        for i in range(self.M):
            if self.verbose and i % 100 == 0:
                print(f"Ran up to t = {i*self.dt}")
                # self.plot_cells()
                # plt.show()
            
            t = i * self.dt
            self.locations = (self.locations+self.velocities * self.dt)%self.L
            self.location_data[i] = self.locations
            self.update_states()
            self.update_velocities()
            self.state_data[i] = self.states
            self.signal_data[i] = self.signal
            self.velocity_data[i] = self.velocities

        return self.location_data, self.state_data, self.velocity_data, self.signal_data
    
    def update_states(self):
        self.signal = self.chemical(self.locations[:,0],self.locations[:,1])
        self.states += self.state_ODE(self.signal,self.states[:,0],self.states[:,1])*self.dt
        self.lambdas = self.lambda0-self.taxis_strength*(self.signal-self.states[:,1])
    def rotation_kernel(self,theta):
        return theta+np.random.uniform(-np.pi,np.pi,self.turn)

    def update_velocities(self):
        # TODO profile this
        r1 = np.random.uniform(0, 1, self.N)
        for i,theta in enumerate(self.thetas):
            if r1[i]<self.lambdas[i]*self.dt:
                self.thetas[i] = self.rotation_kernel(theta)
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
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.axis("equal")
        ax.axis("off")
        plt.show()


class Analysis:
    def __init__(self,locations,states,velocities,parameters,verbose=False):
        self.parameters = parameters
        self.locations = locations
        self.states = states
        self.velocities = velocities
        self.verbose = verbose

    def plot(self,i):
        fig, ax = plt.subplots()
        speeds = lag.norm(self.velocities, axis=1)
        cmap = cm.get_cmap("coolwarm")
        ax.quiver(
            self.locations[i,:, 0],
            self.locations[i,:, 1],
            self.velocities[i,:, 0],
            self.velocities[i,:, 1],
            # color=cmap(speeds),
            scale=20,
        )
        # ax.quiver(self.locations[:,0],self.locations[:,1],self.velocities[:,0],self.velocities[:,1],cmap="viridis")
        ax.set_xlim(0, self.parameters["L"])
        ax.set_ylim(0, self.parameters["L"])
        ax.axis("equal")
        ax.axis("off")
    
    def plot_density(self,i,ax=None,zmax=50):
        locations = self.locations[i,:,:]
        if self.verbose:
            print("Solving for histogram ... ")
        xedges = np.linspace(0,self.parameters["L"],21)
        H,xedges,yedges = np.histogram2d(locations[:,0],locations[:,1],bins=(xedges, xedges))
        if self.verbose:
            print("Plotting ... ")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.set(xlim=(0, self.parameters["L"]), ylim=(0, self.parameters["L"]),zlim=(0, zmax))
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        ax.plot_surface(X, Y, H, cmap=cm.coolwarm,vmin=0,vmax=zmax)
        return ax
    
    def animate(self,stride=1):
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        cmap = cm.get_cmap("jet")
        self.speeds = lag.norm(self.velocities[0],axis=1)
        quiver = ax.quiver(self.locations[0,:,0],self.locations[0,:,1],self.velocities[0,:,0],self.velocities[0,:,1],scale=20,cmap="coolwarm")
        quiver.set_UVC(self.velocities[0,:,0],self.velocities[0,:,1],self.speeds)
        colors = cmap((self.speeds-self.speeds.min())/(self.speeds.max()-self.speeds.min()))
        # ax.set_prop_cycle('color',colors)
        # plot the trajectories, coloring each line based on the speed of the cell
        lines,= ax.plot(self.locations[0,0,0],self.locations[0,0,1])
        ax.axis("off")
        ax.axis("equal")
        ax.set(xlim=(0,self.parameters["L"]),ylim=(0,self.parameters["L"]))
        def update(j):
            lines.set_data(self.locations[:j*stride,0,0],self.locations[:j*stride,0,1])
            i = j*stride
            quiver.set_offsets(self.locations[i])
            self.speeds = lag.norm(self.velocities[i],axis=1)
            quiver.set_UVC(self.velocities[i,:,0],self.velocities[i,:,1],self.speeds)
        anim = animation.FuncAnimation(fig,update,frames=self.locations.shape[0]//stride,interval=10)
        return anim

    def animate_dots(self,stride=1):
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        self.speeds = lag.norm(self.velocities[0],axis=1)
        scatter = ax.scatter(self.locations[0,:,0],self.locations[0,:,1],c="k",s=0.5)
        X,Y = np.meshgrid(np.linspace(0,self.parameters["L"],100),np.linspace(0,self.parameters["L"],100))
        chemical_function = self.parameters["chemical"]
        ax.imshow(chemical_function(X,Y),extent=(0,self.parameters["L"],0,self.parameters["L"]),origin="lower",cmap="coolwarm",alpha=0.5)
        ax.axis("off")
        ax.axis("equal")
        ax.set(xlim=(0,self.parameters["L"]),ylim=(0,self.parameters["L"]))
        def update(j):
            i = j*stride
            scatter.set_offsets(self.locations[i])
        anim = animation.FuncAnimation(fig,update,frames=self.locations.shape[0]//stride,interval=10)
        return anim

    def animate_density(self,stride=1):
        fig = plt.figure()
        xedges = np.linspace(0,self.parameters["L"],21)
        H,xedges,yedges = np.histogram2d(self.locations[-1,:,0],self.locations[-1,:,1],bins=(xedges, xedges))
        zmax = H.max()
        print(zmax)
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(6,6)
        ax1 = self.plot_density(0,ax,zmax)
        def update(j):
            i = j*stride
            ax.clear()
            ax1 = self.plot_density(i,ax,zmax)
        anim = animation.FuncAnimation(fig,update,frames=self.locations.shape[0]//stride,interval=10)
        return anim
