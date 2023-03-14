import numpy as np 
import numpy.linalg as lag
import matplotlib.pyplot as plt

class PDE_Solver: 
    def __init__(self,parameters):
        """Solves the parabolic PDE for the chemical and cell concentrations."""
        self.parameters = parameters
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

