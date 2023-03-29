from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from default_parameters import *
from pyramid import offset_pyramid

parameters["N"] = int(1e4)
parameters["M"] = 2000
parameters["seed"] = 101
parameters["initial_condition"] = "delta"
parameters["Lx"] = 20
parameters["Ly"] = 10
parameters["chemical"] = lambda x,y: 8*offset_pyramid(x,y,c=[12,6])

c = Colony(parameters,False)
foldername = f"data/N={parameters['N']}_params_6.1"
with mp.Pool(mp.cpu_count()) as pool:
    results = pool.starmap_async(c.run,[(True,False,100,foldername,"locations")])
    c.run(overwrite=True,array=False,save_frequency=100,foldername=foldername,save="locations")
a = Analysis(foldername,parameters,verbose=False,stride=1)