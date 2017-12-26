import numpy as np
import Particle_Simulation.Errors as Er
from numba import jitclass
from numba import float64, int64, double

specs = [
    ('particle_positions', float64[:,:]),
    ('box_space', float64[:]),
    ('cutoff', float64),
    ('particle_number', int64),
    ('dim', int64),
    ('cell_number', float64[:]),
    ('cell_space', float64[:]),
    ('total_cell_number', int64),
    ('cell_list', float64[:]),
    ('particle_neighbour_list', float64[:]),
]

@jitclass(specs)
class System:
    def __init__(self, particles, Box, rc):

        self.particle_positions = particles.astype(np.float64)
        self.box_space = Box.astype(np.float64)
        self.cutoff = rc

        self.particle_number = len(particles)

        self.dim = len(self.particle_positions[0])
        self.cell_number = np.zeros(self.dim)
        self.cell_space = np.zeros(self.dim)

        for i in range(len(self.box_space)):
            self.cell_number[i]=(np.floor(self.box_space[i] / self.cutoff))
            self.cell_space[i]=((self.box_space[i] / self.cell_number[i]))

        self.total_cell_number = int(np.prod(self.cell_number))
        self.cell_list = np.zeros(self.total_cell_number) - 1
        self.particle_neighbour_list = np.zeros(self.particle_number) -1

    def update_neighbourlist(self):
        raise NotImplementedError()

    def construct_neighborlist(self):

        for i in range(self.particle_number):
            if (self.dim != len(self.particle_positions[i])):
                raise Er.InputError('Different Dimensions in Particles are not allowed!')

            particle_cell_location = []
            for a in range(len(self.particle_positions[i])):
                if self.particle_positions[i][a] >= self.box_space[a]:
                    self.particle_positions[i][a] -= self.box_space[a]
                particle_cell_location.append(np.floor(self.particle_positions[i][a] / self.cell_space[a]))

            cell_index = self.calculate_index(particle_cell_location)
            self.particle_neighbour_list[i] = self.cell_list[cell_index]
            self.cell_list[cell_index] = i

    def calculate_index(self, particle_cell_location):
        cell_index = 0
        k = self.dim - 1
        j = self.dim
        while k >= 0:
            cell_index += int(particle_cell_location[k] * np.prod(self.cell_number[j:3]))
            j = j - 1
            k = k - 1
        return cell_index

    '''
    Old Code - Don't delete till v1.0
    
    def construct_3dneighborlist(self):
        self.cellnumber = np.floor(np.array([self.Box[0]/self.rc,self.Box[1]/self.rc,self.Box[2]/self.rc]))
        self.cellspace = np.array([self.Box[0]/self.cellnumber[0],self.Box[1]/self.cellnumber[1],
                                   self.Box[2]/self.cellnumber[2]])
        self.totalcellnumber = int(np.prod(self.cellnumber))
        self.head = np.zeros(self.totalcellnumber)-1
        self.list = np.zeros(self.particlenumber)-1
        self.mc = np.zeros(3)

        for i in range(self.particlenumber):
            for a in range(0,2):
                self.mc[a] = np.floor(self.particles[i][a] / self.cellspace[a])
            index = int(self.mc[2] + self.mc[1]*self.cellnumber[2] + self.mc[0]*self.cellnumber[1]*self.cellnumber[2])
            self.list[i] = self.head[index]
            self.head[index] = i
    
    self.cell_number = np.ones(3)  
    self.cell_space = np.zeros(3)

    cell_index = int(particle_cell_location[2] + particle_cell_location[1] * self.cell_number[2] +
        particle_cell_location[0] * self.cell_number[2] * self.cell_number[1])
    '''
