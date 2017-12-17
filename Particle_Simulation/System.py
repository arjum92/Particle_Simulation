import numpy as np
import Particle_Simulation.Errors as Er

'''
Boundary Conditions!!
Can we use it this way?
'''


class System:
    def __init__(self, particles, Box, rc):

        self.particles = particles
        self.Box = Box
        self.rc = rc

        self.particlenumber = len(particles)
        self.cellnumber = np.ones(3)
        self.cellspace = np.ones(3)

        self.dim = len(self.particles[0])

    def update_neighborlist(self):
        raise NotImplementedError()

    def construct_neighborlist(self):
        for i in range(len(self.Box)):
            self.cellnumber[i] = np.floor(self.Box[i] / self.rc)
            self.cellspace[i] = self.Box[i] / self.cellnumber[i]

            self.totalcellnumber = int(np.prod(self.cellnumber))
            self.head = np.zeros(self.totalcellnumber) - 1
            self.list = np.zeros(self.particlenumber) - 1

            mc = np.zeros(3)

        for i in range(self.particlenumber):
            if(self.dim != len(self.particles[i])):
                raise Er.InputError('Different Dimensions in Particles are not allowed!')

            for a in range(len(self.particles[i])):
                mc[a] = np.floor(self.runden(self.particles[i][a] / self.cellspace[a]))

            index = int(mc[2] + mc[1] * self.cellnumber[2] + mc[0] * self.cellnumber[2] * self.cellnumber[1])

            self.list[i] = self.head[index]
            self.head[index] = i

    def runden(self, x):
        if x > 0 and x % 1 == 0:
            x = x - 0.1
        return x

    '''
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
    '''
