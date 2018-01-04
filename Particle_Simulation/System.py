from Particle_Simulation.Energy import Energy
from Particle_Simulation.Neighbourlist import Neighbourlist


class System:
    cell_neighbour_list = None

    def __init__(self, particles, paramenter):
        self.particles = particles
        self.energy = Energy()
        self.neighbourlist = Neighbourlist(Box=paramenter.box, rc=paramenter.cutoff_radius, particles=particles)

        if System.cell_neighbour_list == None:
            System.cell_neighbour_list = self.neighbourlist.calc_cell_neighbours()
        self.k_vectors = [1,1,1]