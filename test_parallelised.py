from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from default_parameters import *
from pyramid import pyramid,offset_pyramid
import os
import shutil as sh
import time

def worker(subcolony):
    return subcolony.run()

def run_parallel(parameters,foldername,n_cores=mp.cpu_count):
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
        # os.mkdir(f"{foldername}/subcolony_{i}")
        subcolony_parameters[i]["N"] = population_sizes[i]
        subcolony_parameters[i]["seed"] = parameters["seed"]+i
        subcolony_parameters[i]["foldername"] = f"{foldername}/subcolony_{i}"
        subcolonies.append(Colony(subcolony_parameters[i],False))
    subcolonies[0].verbose = True
    print('Starting parallel run with {} cores'.format(n_cores))
    p = mp.Pool(n_cores)
    results = p.map(worker,subcolonies)
    p.close()
    for result in results:
        print(result.shape)
    return np.concatenate(results,axis=1)


def state_ODE(C,y1,y2):
    return np.array([(C-y1-y2)/parameters["te"],(C-y2)/parameters["ta"]]).T
def chemical(x,y):
    return 8*offset_pyramid(x,y,c=[12,6])

if __name__ == "__main__":

    parameters = {
        "M": 50000,
        "N": 10000,
        "Lx": 20,
        "Ly": 10,
        "te": 0.001,
        "ta": 1,
        "dt": 0.01,
        "dx": 0.01,
        "chemical": chemical,
        "taxis_strength": 1,
        "speed": 0.1,
        "n_states":2,
        "state_ODE": state_ODE,
        "lambda0": 1,
        "initial_condition": "delta",
        "dimension": 2,
        "seed": 4,
        "overwrite":False,
        "array":True,
        "save_frequency":100,
        "foldername":"test",
        "save":"none"
    }

    foldername = f"data/N={parameters['N']}_params_6.1_y1"
    times = np.arange(0,parameters["dt"]*parameters["M"],parameters["dt"]*parameters["save_frequency"])
    n_cores = 8
    locations = run_parallel(parameters,foldername,n_cores=n_cores)
    for i,time in enumerate(times):
        np.savez(f"{foldername}/data_{np.round(time,2)}",locations=locations[i])
    a = Analysis(foldername,parameters,verbose=False,stride=1)
    anim = a.animate_dots()
    anim.save(f"media/weird_dots_small_N={parameters['N']}_{n_cores}.mp4",fps=40)

    # timing the functions
    # start_8 = time.perf_counter()
    # locations = run_parallel(parameters,foldername,n_cores=8)
    # end_8 = time.perf_counter()
    # print(f"8 cores took {end_8-start_8} seconds")
    # start_1 = time.perf_counter()
    # locations = run_parallel(parameters,foldername,n_cores=1)
    # end_1 = time.perf_counter()
    # print(f"1 core took {end_1-start_1} seconds")