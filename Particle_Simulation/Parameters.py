import numpy as np


class Parameters:
    VACUUM_PERMITTIVITY = 1
    BOLTZMANN_CONSTANT = 1

    cell_shift_list = np.array([
        [0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0],
        [0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])


    def __init__(self):
        self.temperature = None
        self.es_sigma = None
        self.mc_update_radius = None
        self.particle_types = None

    def __init__(self, temperature, box, es_sigma, update_radius, particle_types, cutoff_radius):
        self.temperature = temperature
        self.box = box
        self.es_sigma = es_sigma
        self.update_radius = update_radius
        self.particle_types = particle_types
        self.cutoff_radius = cutoff_radius
