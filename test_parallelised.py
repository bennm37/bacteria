from solver import Colony
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from default_parameters import *
from pyramid import pyramid, offset_pyramid
import os
import shutil as sh
import time


def worker(subcolony):
    return subcolony.run()


def run_parallel(parameters, foldername, n_cores=mp.cpu_count):
    popultation_indicies = np.array_split(np.arange(parameters["N"]), n_cores)
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
        elif input(f"Delete Folder {foldername}?    ").lower() in ["y", "yes"]:
            sh.rmtree(foldername)
            os.mkdir(foldername)
        else:
            raise ValueError("Folder already exists")
    for i in range(n_cores):
        # os.mkdir(f"{foldername}/subcolony_{i}")
        subcolony_parameters[i]["N"] = population_sizes[i]
        subcolony_parameters[i]["seed"] = parameters["seed"] + i
        subcolony_parameters[i]["foldername"] = f"{foldername}/subcolony_{i}"
        subcolonies.append(Colony(subcolony_parameters[i], False))
    subcolonies[0].verbose = True
    print("Starting parallel run with {} cores".format(n_cores))
    p = mp.Pool(n_cores)
    results = p.map(worker, subcolonies)
    p.close()
    for result in results:
        print(result.shape)
    return np.concatenate(results, axis=1)


sins = lambda x, y: 5*np.sin(5 * np.pi * x / 10) * np.sin(5 * np.pi * y / 10)
gaussian = lambda x, y: 10*np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 2)

def plot_sin_gaussian_intial():
    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax0.plot_surface(X, Y, sins(X, Y),cmap="plasma")
    ax1.plot_surface(X, Y, gaussian(X, Y),cmap="plasma")
    plt.show()

def state_ODE_sin_gaussian(C, y, te, ta):
    return np.array(
        [
            (C[:, 0] - y[:, 0]) / ta[0],
            (C[:, 1] - y[:, 1]) / ta[1],
            (np.sum(C, axis=1) - np.sum(y)) / te,
        ]
    ).T

def chemical_sin_gaussian(x, y):
    return np.array([sins(x, y), gaussian(x, y)]).T


def run_sin_gaussian():
    parameters = {
        "M": 200000,
        "N": int(1e6),
        "Lx": 10,
        "Ly": 10,
        "te": 0.001,
        "ta": np.array([3,1]),
        "dt": 0.01,
        "dx": 0.01,
        "chemical": chemical_sin_gaussian,
        "taxis_strength": 1,
        "speed": 0.1,
        "n_states": 3,
        "state_ODE": state_ODE_sin_gaussian,
        "lambda0": 1,
        "initial_condition": "uniform",
        "dimension": 2,
        "seed": 4,
        "overwrite": False,
        "array": True,
        "save_frequency": 100,
        "foldername": "data/N=10000_params_6.1_y1",
        "save": "none",
    }
    foldername = f"data/N={parameters['N']}_sin_gaussian_ta_{parameters['ta'][0]}_{parameters['ta'][1]}"
    times = np.arange(
        0,
        parameters["dt"] * parameters["M"],
        parameters["dt"] * parameters["save_frequency"],
    )
    n_cores = 8
    locations = run_parallel(parameters,foldername,n_cores=n_cores)
    # c = Colony(parameters, verbose=True)
    # locations = c.run()
    for i, time in enumerate(times):
        np.savez(f"{foldername}/data_{np.round(time,2)}", locations=locations[i])
    return foldername,parameters

def plot_two_peaks_initial():
    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax0.plot_surface(X, Y, gaussian(X-2,Y),cmap="plasma")
    ax1.plot_surface(X, Y, gaussian(X+2, Y),cmap="plasma")
    plt.show()

def state_ODE_two_peaks(C, y, te, ta):
    return np.array(
        [
            (C[:, 0] - y[:, 0]) / ta[0],
            (C[:, 1] - y[:, 1]) / ta[1],
            (np.sum(C, axis=1) - np.sum(y)) / te,
        ]
    ).T

def chemical_two_peaks(x, y):
    return np.array([gaussian(x-2,y), gaussian(x+2, y)]).T

def run_two_peaks():
    parameters = {
        "M": 200000,
        "N": int(1e6),
        "Lx": 10,
        "Ly": 10,
        "te": 0.001,
        "ta": np.array([3,1]),
        "dt": 0.01,
        "dx": 0.01,
        "chemical": chemical_two_peaks,
        "taxis_strength": 1,
        "speed": 0.1,
        "n_states": 3,
        "state_ODE": state_ODE_two_peaks,
        "lambda0": 1,
        "initial_condition": "uniform",
        "dimension": 2,
        "seed": 4,
        "overwrite": False,
        "array": True,
        "save_frequency": 100,
        "foldername": "data/N=10000_params_6.1_y1",
        "save": "none",
    }
    foldername = f"data/N={parameters['N']}_two_peaks_ta_{parameters['ta'][0]}_{parameters['ta'][1]}"
    times = np.arange(
        0,
        parameters["dt"] * parameters["M"],
        parameters["dt"] * parameters["save_frequency"],
    )
    n_cores = 8
    locations = run_parallel(parameters,foldername,n_cores=n_cores)
    # c = Colony(parameters, verbose=True)
    # locations = c.run()
    for i, time in enumerate(times):
        np.savez(f"{foldername}/data_{np.round(time,2)}", locations=locations[i])
    return foldername,parameters

def state_ODE_6_1(C, y):
    # confusingly, y1 from erban paper is y[1] here, and y2 is y[0]
    # this is consistent with the notation used in my special topic
    return np.array(
        [(C - y[:, 1] - y[:, 0]) / parameters["te"], (C - y[:, 0]) / parameters["ta"]]
    ).T


def chemical_6_1(x, y):
    return 8 * offset_pyramid(x, y, c=[12, 6])


def run_figure_6_1():
    parameters = {
        "M": 5000,
        "N": 1000,
        "Lx": 20,
        "Ly": 10,
        "te": 0.001,
        "ta": 1,
        "dt": 0.01,
        "dx": 0.01,
        "chemical": chemical_6_1,
        "taxis_strength": 1,
        "speed": 0.1,
        "n_states": 2,
        "state_ODE": state_ODE_6_1,
        "lambda0": 1,
        "initial_condition": "delta",
        "dimension": 2,
        "seed": 4,
        "overwrite": False,
        "array": True,
        "save_frequency": 100,
        "foldername": "test",
        "save": "none",
    }

    foldername = f"data/N={parameters['N']}_test_multiple"
    times = np.arange(
        0,
        parameters["dt"] * parameters["M"],
        parameters["dt"] * parameters["save_frequency"],
    )
    n_cores = 8
    locations = run_parallel(parameters, foldername, n_cores=n_cores)
    for i, time in enumerate(times):
        np.savez(f"{foldername}/data_{np.round(time,2)}", locations=locations[i])
    return foldername,parameters


if __name__ == "__main__":
    # run_figure_6_1()
    foldername1,parameters1 = run_two_peaks()
    foldername2,parameters2 = run_sin_gaussian()
    plot_two_peaks_initial()
    a = Analysis(foldername1, parameters1, verbose=False, stride=1)
    anim = a.animate_density(10, nedges=40, scale=10)
    plt.show()
    # timing the functions
    # start_8 = time.perf_counter()
    # locations = run_parallel(parameters,foldername,n_cores=8)
    # end_8 = time.perf_counter()
    # print(f"8 cores took {end_8-start_8} seconds")
    # start_1 = time.perf_counter()
    # locations = run_parallel(parameters,foldername,n_cores=1)
    # end_1 = time.perf_counter()
    # print(f"1 core took {end_1-start_1} seconds")
