import numpy as np 
import numpy.linalg as lag
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import os
import shutil as sh
import glob

class Analysis:
    def __init__(self,foldername,parameters,verbose=False,stride=1):
        self.parameters = parameters
        self.foldername = foldername
        self.verbose = verbose
        self.load_data(stride=stride)
        if verbose:
            print("Loaded Data")
    
    def load_data(self,stride = 1):
        npzfiles = glob.glob(f"{self.foldername}/*.npz")
        self.times = [float(file.split("_")[-1][:-4]) for file in npzfiles]
        # sort files based on times 
        npzfiles = [file for _,file in sorted(zip(self.times,npzfiles))]
        n_files = len(npzfiles)//stride
        self.location_data = np.zeros([n_files,self.parameters["N"],self.parameters["dimension"]])
        self.state_data = np.zeros([n_files,self.parameters["N"],self.parameters["n_states"]])
        self.velocity_data = np.zeros([n_files,self.parameters["N"],self.parameters["dimension"]])
        self.signal_data = np.zeros([n_files,self.parameters["N"]])
        for i,file in enumerate(npzfiles[::stride]):
            data = np.load(file)
            self.location_data[i] = data["locations"]
            self.state_data[i] = data["states"]
            self.velocity_data[i] = data["velocities"]
            self.signal_data[i] = data["signal"]

    def plot(self,i):
        fig, ax = plt.subplots()
        speeds = lag.norm(self.velocity_data, axis=1)
        cmap = cm.get_cmap("coolwarm")
        ax.quiver(
            self.location_data[i,:, 0],
            self.location_data[i,:, 1],
            self.velocity_data[i,:, 0],
            self.velocity_data[i,:, 1],
            # color=cmap(speeds),
            scale=20,
        )
        # ax.quiver(self.location_data[:,0],self.location_data[:,1],self.velocity_data[:,0],self.velocity_data[:,1],cmap="viridis")
        ax.set_xlim(0, self.parameters["L"])
        ax.set_ylim(0, self.parameters["L"])
        ax.axis("equal")
        ax.axis("off")
    
    def plot_density(self,i,ax=None,zmax=50,nedges=21):
        locations = self.location_data[i,:,:]
        if self.verbose:
            print("Solving for histogram ... ")
        xedges = np.linspace(0,self.parameters["L"],nedges)
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
        self.speeds = lag.norm(self.velocity_data[0],axis=1)
        quiver = ax.quiver(self.location_data[0,:,0],self.location_data[0,:,1],self.velocity_data[0,:,0],self.velocity_data[0,:,1],scale=20,cmap="coolwarm")
        quiver.set_UVC(self.velocity_data[0,:,0],self.velocity_data[0,:,1],self.speeds)
        colors = cmap((self.speeds-self.speeds.min())/(self.speeds.max()-self.speeds.min()))
        # ax.set_prop_cycle('color',colors)
        # plot the trajectories, coloring each line based on the speed of the cell
        lines,= ax.plot(self.location_data[0,0,0],self.location_data[0,0,1])
        ax.axis("off")
        ax.axis("equal")
        ax.set(xlim=(0,self.parameters["L"]),ylim=(0,self.parameters["L"]))
        def update(j):
            lines.set_data(self.location_data[:j*stride,0,0],self.location_data[:j*stride,0,1])
            i = j*stride
            quiver.set_offsets(self.location_data[i])
            self.speeds = lag.norm(self.velocity_data[i],axis=1)
            quiver.set_UVC(self.velocity_data[i,:,0],self.velocity_data[i,:,1],self.speeds)
        anim = animation.FuncAnimation(fig,update,frames=self.location_data.shape[0]//stride,interval=10)
        return anim

    def animate_dots(self,stride=1):
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        self.speeds = lag.norm(self.velocity_data[0],axis=1)
        scatter = ax.scatter(self.location_data[0,:,0],self.location_data[0,:,1],c="k",s=0.5)
        X,Y = np.meshgrid(np.linspace(0,self.parameters["L"],100),np.linspace(0,self.parameters["L"],100))
        chemical_function = self.parameters["chemical"]
        ax.imshow(chemical_function(X,Y),extent=(0,self.parameters["L"],0,self.parameters["L"]),origin="lower",cmap="coolwarm",alpha=0.5)
        ax.axis("off")
        ax.axis("equal")
        ax.set(xlim=(0,self.parameters["L"]),ylim=(0,self.parameters["L"]))
        def update(j):
            i = j*stride
            scatter.set_offsets(self.location_data[i])
        anim = animation.FuncAnimation(fig,update,frames=self.location_data.shape[0]//stride,interval=10)
        return anim

    def animate_density(self,stride=1,nedges=21):
        fig = plt.figure()
        xedges = np.linspace(0,self.parameters["L"],nedges)
        H,xedges,yedges = np.histogram2d(self.location_data[-1,:,0],self.location_data[-1,:,1],bins=(xedges, xedges))
        zmax = H.max()
        print(zmax)
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(6,6)
        ax1 = self.plot_density(0,ax,zmax=zmax,nedges=nedges)
        def update(j):
            i = j*stride
            ax.clear()
            ax1 = self.plot_density(i,ax,zmax,nedges=nedges)
        anim = animation.FuncAnimation(fig,update,frames=self.location_data.shape[0]//stride,interval=10)
        return anim
