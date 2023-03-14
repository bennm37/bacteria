import cProfile
import pstats

def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        stats = pstats.Stats(pr).sort_stats('tottime')
        stats.print_stats(10)
        return retval
    return inner

@profile
def run_simulation():
    from solver import Colony,Analysis
    import matplotlib.pyplot as plt
    import numpy as np
    from default_parameters import parameters
    parameters["N"] = 10000
    parameters
    c = Colony(parameters,False)
    location_data,state_data,velocity_data,signal_data = c.run()

run_simulation()