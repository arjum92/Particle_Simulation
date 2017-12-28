import numpy as np
import unittest
from Particle_Simulation.Neighbourlist import Neighbourlist

'''
Pls write true test-classes
'''


class test_System(unittest.TestCase):
    def test_1d(self):
        particle_positions = np.array([[1], [0], [3], [4], [0]])
        box_space = np.array([4])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_2d(self):
        particle_positions = np.array([[1, 1], [0, 3], [3, 4], [4, 4], [0, 0]])
        box_space = np.array([4, 4])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_2dsecond(self):
        particle_positions = np.array(
            [[0.01, 0.01], [0, 0.03], [0.03, 0.04], [0.05, 0.05], [0.09, 0.09], [0.09, 0], [0.1, 0.1]])
        box_space = np.array([0.10, 0.10])
        cutoff = 0.025
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.particle_neighbour_list)
        print(s1.cell_list)

    def test_3d(self):
        particle_positions = np.array([[1, 1, 3], [0, 3, 1], [3, 4, 2], [4, 4, 4], [0, 0, 0]])
        box_space = np.array([4, 4, 4])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_3dsecond(self):
        particle_positions = np.array([[9, 6, 4], [2.25, 2, 2], [2.3, 4, 1], [9, 4, 2], [0, 0, 0]])
        box_space = np.array([9, 6, 4])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    # jit is only better, if we have 100.000 particles
    def test_Bench(self):
        particle_positions = []
        for i in range(0, 100000):
            xtest = np.random.rand(2)
            particle_positions.append(xtest)
        particle_positions = np.array(particle_positions)
        box_space = np.array([1, 1])
        cutoff = 0.1
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_periodicity(self):
        particle_positions = np.array([[19, -18]], dtype=np.int32)
        box_space = np.array([2, 2], dtype=np.int32)
        cutoff = 1
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_list(self):
        particle_positions = np.array([[19, -18]])
        box_space = np.array([4, 4])
        cutoff = 1
        s1 = Neighbourlist(particle_positions, box_space, cutoff)

        print(s1.cell_neighbour_list_2D())

    def test_list1d(self):
        particle_positions = np.array([[1], [0], [3], [4], [0]])
        box_space = np.array([4])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        print(s1.cell_neighbour_list_1D())

    def test_list3d(self):
        particle_positions = np.array([[9, 6, 4]])
        box_space = np.array([8, 6, 4])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        print(s1.calculate_index(np.array([1, 0, 0])))
        lol = s1.cell_neighbour_list_3D()
        print(lol)

    def test_benchmark(self):
        particle_positions = []
        for i in range(0, 10000000):
            xtest = np.random.rand(3)
            particle_positions.append(xtest)
        particle_positions = np.array(particle_positions)
        box_space = np.array([500, 500, 500])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        print(s1.calculate_index(np.array([1, 0, 0])))
        lol = s1.cell_neighbour_list_3D()
        s1.construct_neighborlist()
        print(lol)
        print(s1.cell_list)
        print(s1.particle_neighbour_list)


'''
Benchmark:

Jit only sensefull with over 100.000 particles or cells

10.000.000 Particles 15 seconds calculating time and 1 GB RAM for head and list
15.625.000 cells (250x250x250) 90 seconds calculating time for cell_nl and 4 GB for matrix

both 105 seconds and 4.1 GB RAM

'''
