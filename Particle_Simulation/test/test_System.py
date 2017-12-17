import numpy as np
import unittest
from Particle_Simulation.System import System

'''
Pls write true test-classes
'''

class test_System(unittest.TestCase):

    def test_1d(self):
        print(np.floor(3.6 / 4))
        particles = np.array([[1], [0], [3], [4], [0]])
        Box = np.array([4])
        rc = 2
        s1 = System(particles, Box, rc)
        s1.construct_neighborlist()
        print(s1.head)
        print(s1.list)

    def test_2d(self):
        print(np.floor(3.6 / 4))
        particles = np.array([[1, 1], [0, 3], [3, 4], [4, 4], [0, 0]])
        Box = np.array([4, 4])
        rc = 2
        s1 = System(particles, Box, rc)
        s1.construct_neighborlist()
        print(s1.head)
        print(s1.list)

    def test_3d(self):
        print(np.floor(3.6 / 4))
        particles = np.array([[1, 1, 3], [0, 3, 1], [3, 4, 2], [4, 4, 4], [0, 0, 0]])
        Box = np.array([4, 4, 4])
        rc = 2
        s1 = System(particles, Box, rc)
        s1.construct_neighborlist()
        print(s1.head)
        print(s1.list)
