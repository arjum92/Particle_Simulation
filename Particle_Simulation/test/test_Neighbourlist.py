import numpy as np
import unittest
import numpy.testing as npt
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

    @unittest.expectedFailure
    def test_dimension_mismatch(self):
        # particle_position_dimension_diff_with_box
        particle_positions = np.array([[0.0, 0.5, 3], [1.5, 4.5], [2.5, 2.5],
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1],
                                       [8.25, 2.0], [9.75, 11.0], [10.0, 8.0]])
        box_space = np.array([10.5, 2.0])
        cutoff = 2
        npt.assert_raises(ValueError, Neighbourlist, particle_positions, box_space, cutoff)

    @unittest.expectedFailure
    def test_null_value(self):
        # particle_position_dimension_diff_with_box
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5],
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1],
                                       [8.25, 2.0], [9.75, 11.0], []])
        box_space = np.array([10.5, 11.0])
        cutoff = 4.6
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()

    @unittest.expectedFailure
    def test_string_value(self):
        # particle_position_dimension_diff_with_box
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5],
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1],
                                       [8.25, 2.0], [9.75, 11.0], ['h', 'm']])
        box_space = np.array([10.5, 11.0])
        cutoff = 4.6
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()

    def test_1d_output_validation_1(self):
        particle_positions = np.array([[1.5], [2.5],
                                       [3.0], [4.5], [5.5],
                                       [6.0], [6.5], [7.0],
                                       [8.25], [9.75], [10.0]])
        reference_head = np.asarray([10.0, 6.0, 9.0])
        reference_neighlist = np.asarray([-1.0, 0.0, 1.0, -1.0, 3.0, 4.0, 5.0, -1.0, 7.0, 8.0, 2.0])
        box_space = np.array([10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_1d_output_validation_2(self):
        particle_positions = np.array([[1.5], [2.5],
                                       [3.0], [4.5], [5.5],
                                       [6.0], [6.5], [7.0],
                                       [8.25], [9.75], [10.0], [11.5], [12.5]])
        reference_head = np.asarray([12.0, 6.0, 9.0])
        reference_neighlist = np.asarray([-1.0, 0.0, 1.0, -1.0, 3.0, 4.0, 5.0, -1.0, 7.0, 8.0, 2.0, 10.0, 11.0])
        box_space = np.array([10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_1d_output_validation_3(self):
        particle_positions = np.array([[-1.75], [-0.25], [1.5], [2.5],
                                       [3.0], [4.5], [5.5],
                                       [6.0], [6.5], [7.0],
                                       [8.25], [9.75], [10.0], [11.5], [12.5]])
        reference_head = np.asarray([14.0, 8.0, 11.0])
        reference_neighlist = np.asarray(
            [-1.0, 0.0, -1.0, 2.0, 3.0, -1.0, 5.0, 6.0, 7.0, 1.0, 9.0, 10.0, 4.0, 12.0, 13.0])
        box_space = np.array([10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_3d_output_validation(self):
        particle_positions = np.array([[1.5, 1., 2.5], [1.5, 4., 2.5], [1.5, 8.5, 2.5],
                                       [1.5, 1., 5.5], [1.5, 4., 5.5], [1.5, 5.5, 5.5],
                                       [1.5, 10., 5.5], [1.5, 1., 8.5], [1.5, 4., 8.5],
                                       [1.5, 8.5, 8.5], [1.5, 4., 11.5], [1.5, 11.5, 11.5],
                                       [4.5, 1., 2.5], [4.5, 5.5, 2.5], [4.5, 7., 2.5],
                                       [4.5, 1., 5.5], [4.5, 4., 5.5], [4.5, 5.5, 5.5],
                                       [4.5, 7., 5.5], [4.5, 10., 5.5], [4.5, 2.5, 8.5],
                                       [4.5, 4., 8.5], [4.5, 5.5, 8.5], [4.5, 8.5, 8.5],
                                       [4.5, 1., 11.5], [4.5, 5.5, 11.5], [7.5, 1., 2.5],
                                       [7.5, 5.5, 2.5], [7.5, 7., 2.5], [7.5, 11.5, 2.5],
                                       [7.5, 1., 5.5], [7.5, 4., 5.5], [7.5, 5.5, 5.5],
                                       [7.5, 10., 5.5], [7.5, 2.5, 8.5], [7.5, 5.5, 8.5],
                                       [7.5, 7., 8.5], [7.5, 11.5, 8.5], [7.5, 4., 11.5],
                                       [7.5, 7., 11.5], [10.5, 1., 2.5], [10.5, 5.5, 2.5],
                                       [10.5, 11.5, 2.5], [10.5, 1., 5.5], [10.5, 2.5, 5.5],
                                       [10.5, 4., 5.5], [10.5, 10., 5.5], [10.5, 11.5, 5.5],
                                       [10.5, 1., 8.5], [10.5, 4., 8.5], [10.5, 11.5, 8.5],
                                       [10.5, 1., 11.5], [10.5, 4., 11.5], [10.5, 11.5, 11.5],
                                       [3., 3.25, 3.5], [6.0, 6.5, 7.0], [6.5, 5.0, 7.0],
                                       [8.0, 6.5, 7.0], [12.0, 6.5, 7.0], [12.0, 6.5, 3.5],
                                       [12.0, 13.0, 7.0], [6.0, 13.0, 7.0], [13.5, 1., 5.5],
                                       [13.5, 4., 5.5], [13.5, 5.5, 5.5], [13.5, 10., 5.5],
                                       [13.5, 14, 5.5], [13.5, 15.5, 5.5], [-1.5, 1., 5.5],
                                       [-1.5, 2.5, 5.5], [-1.5, 4., 5.5], [-1.5, 10., 5.5],
                                       [-1.5, 11.5, 5.5]])

        box_space = np.array([12., 13., 14.])
        reference_head = np.asarray([0, 67, 60, -1, 1, 64, 8, 10, 2, 59, 58, -1, -1, 65, -1, 11,
                                     12, 15, 20, 24, 13, 54, 22, 25, 14, 18, 23, -1, -1, 19, -1,
                                     - 1, 26, 30, 61, -1, 27, 32, 56, 38, 28, -1, 57, 39, 29, 33,
                                     37, -1, 40, 69, 48, 51, 41, 70, 49, 52, -1, -1, -1, -1, 42,
                                     72, 50, 53])
        reference_neighlist = np.asarray([-1, -1, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1,
                                          - 1, -1, -1, 16, -1, -1, -1, -1, 21, -1, -1, -1, -1, -1,
                                          - 1, -1, -1, -1, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          - 1, -1, 43, -1, -1, 46, -1, -1, -1, -1, -1, -1, 17, 36,
                                          35, 55, 9, -1, 7, 34, 3, 5, 63, 6, 62, 66, 44, 68, 45,
                                          47, 71, ])

        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)


''' Yet to implement
   def test_2d_output_validation(self):
       particle_positions = np.array( [[0.0, 0.0], [10.5,0.0], [0.0, 12.0], 
                                       [10.5, 12], [7, 0], [10.5, 6],
                                       [7.0, 12.0], [0.0,6.0], [3.5, 3.0], 
                                       [7.0, 3.0], [3.5 ,6.0], [7.0,6.0],
                                       [7.0,5.0], [6.0, 6.0], [3.5,6.0],
                                       [8.0,9.0], [1.0,4.0], [4.0,1.0],
                                       [5.0,11.0], [2.0,11.0], [9.0,2.0],
                                       [9.5,4.0], [7.5,7.5], [10.0,10.5],
                                       [3.0,7.0],[2.5,9.0],
                                       [3.5,10.0],[8.0,8.0],[9.0,9.0],
                                       [10.0,10.0],[4.0,3.0] ] )
       box_space = np.array([10.5 , 11.0])
       reference_neighlist = np.asarray([-1., 0., 1., 2., -1., 4., 5., 6., -1., 8., 9., 10.])
       box_space = np.array([10])
       cutoff = 3
       s1 = System(particle_positions, box_space, cutoff)
       s1.construct_neighborlist()
       print(s1.cell_list)
       print(s1.particle_neighbour_list)
       npt.assert_equal(reference_head, s1.cell_list ,'Failed',verbose=True )
       npt.assert_equal(reference_neighlist, s1.particle_neighbour_list ,'Failed',verbose=True )
   

   
'''
'''
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

'''
Benchmark:

Jit only sensefull with over 100.000 particles or cells

10.000.000 Particles 15 seconds calculating time and 1 GB RAM for head and list
15.625.000 cells (250x250x250) 90 seconds calculating time for cell_nl and 4 GB for matrix

both 105 seconds and 4.1 GB RAM

'''
