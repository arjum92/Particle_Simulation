import numpy as np
from Energy import Energy
from Neighbourlist import Neighbourlist


class System:

    cell_neighbour_list = None

    def __init__(self, particles, parameters):
        self.particles = particles
        self.energy = Energy()

        x = []
        for i in range(len(particles)):
            x.append(particles[i].position)
        y = np.array(x)

        self.neighbourlist = Neighbourlist(particle_positions=y, Box=parameters.box, rc=parameters.cutoff_radius)

        if System.cell_neighbour_list is None:
            System.cell_neighbour_list = self.neighbourlist.calc_cell_neighbours()
