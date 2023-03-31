from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from default_parameters import *
from pyramid import offset_pyramid
import os
import shutil as sh

def tidy_up(foldername):
    """Combines all location data file from subcolony folders into 
    one file in the parent folder"""
    pass

def run_parallel(parameters,foldername):
    n_cores = mp.cpu_count()
    popultation_indicies = np.array_split(np.arange(parameters["N"]),n_cores)
    population_sizes = [len(population) for population in popultation_indicies]
    subcolony_parameters = [parameters.copy() for i in range(n_cores)]
    subcolonies = []
    parameters["array"] = True
    parameters["save"] = "none"
    try:
        os.mkdir(foldername)
    except FileExistsError:
        if parameters["overwrite"]:
            sh.rmtree(foldername)
            os.mkdir(foldername)
        elif input(f"Delete Folder {foldername}?    ").lower() in ["y","yes"]:
            sh.rmtree(foldername)
            os.mkdir(foldername)
        else:
            raise ValueError("Folder already exists")
    for i in range(n_cores):
        os.mkdir(f"{foldername}/subcolony_{i}")
        subcolony_parameters[i]["N"] = population_sizes[i]
        subcolony_parameters[i]["seed"] = parameters["seed"]+i
        subcolony_parameters[i]["foldername"] = f"{foldername}/subcolony_{i}"
        subcolonies.append(Colony(subcolony_parameters[i],False))
    def worker(subcolony):
        return subcolony.run()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap(worker,subcolonies)
    for result in results:
        print(result)
    tidy_up(foldername)

if __name__ == "__main__":
    parameters["N"] = int(1e2)
    parameters["M"] = 20
    parameters["seed"] = 101
    parameters["initial_condition"] = "delta"
    parameters["Lx"] = 20
    parameters["Ly"] = 10
    parameters["chemical"] = lambda x,y: 8*offset_pyramid(x,y,c=[12,6])

    c = Colony(parameters,False)
    foldername = f"data/N={parameters['N']}_params_6.1"
    # a = Analysis(foldername,parameters,verbose=False,stride=1)
    run_parallel(parameters,foldername)