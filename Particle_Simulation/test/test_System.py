import numpy as np
import numpy.testing as npt
import unittest
from  Particle_Simulation.System import System

'''
Pls write true test-classes
'''

class test_System(unittest.TestCase):

    def test_1d(self):
        particle_positions = np.array([[1], [0], [3], [4], [0]])
        box_space = np.array([4])
        cutoff = 2
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_2d(self):
        particle_positions = np.array([[1, 1], [0, 3], [3, 4], [4, 4], [0, 0]])
        box_space = np.array([4, 4])
        cutoff = 2
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_2dsecond(self):
        particle_positions = np.array([[0.01, 0.01], [0, 0.03], [0.03, 0.04], [0.05, 0.05], [0.09, 0.09], [0.09, 0], [0.1,0.1]])
        box_space = np.array([0.10, 0.10])
        cutoff = 0.025
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_3d(self):
        particle_positions = np.array([[1, 1, 3], [0, 3, 1], [3, 4, 2], [4, 4, 4], [0, 0, 0]])
        box_space = np.array([4, 4, 4])
        cutoff = 2
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

    def test_3dsecond(self):
        particle_positions = np.array([[9, 6, 4], [2.25, 2, 2], [2.3, 4, 1], [4, 4, 2], [0, 0, 0]])
        box_space = np.array([9, 6, 4])
        cutoff = 2
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
    def test(self):
        array=[4,4]
        print(array[1:3])
    
    #Basic imput values validations
    def test_null_value(self):
        #particle_position_dimension_diff_with_box
        particle_positions = np.array( [[ 0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0], [] ] )
        box_space = np.array([10.5 , 11.0])
        cutoff = 4.6
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        
    
    def test_string_value(self):
        #particle_position_dimension_diff_with_box
        particle_positions = np.array( [[ 0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0], ['h','m'] ] )
        box_space = np.array([10.5 , 11.0])
        cutoff = 4.6
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        
    def test_negative_value(self):
        #particle_position_dimension_diff_with_box
        particle_positions = np.array( [[ -1.0, -0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0] ] )
        box_space = np.array([10.5 , 11.0])
        cutoff = 4.6
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        print('Output is coming for negative inputs')
        
        
    def test_dimension_mismatch(self):
        #particle_position_dimension_diff_with_box
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0], [10.0 ,8.0 ] ])
        box_space = np.array([10.5 , 11.0, 2.0])
        npt.assert_equal(len(particle_positions[0]), len(box_space), 'Fail: dimension of box and particle positions not equal')
        cutoff = 3.5
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        
        
    def test_1d_output_validation(self):
        particle_positions = np.array([[0.0], [1.5], [2.5], 
                                       [3.0], [4.5] , [5.5],
                                       [6.0], [6.5], [7.0], 
                                       [8.25], [9.75], [10.0] ])
        reference_head = np.asarray([3.0, 7.0, 11.0])
        reference_neighlist = np.asarray([-1., 0., 1., 2., -1., 4., 5., 6., -1., 8., 9., 10.])
        box_space = np.array([10])
        cutoff = 3
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)
        npt.assert_equal(reference_head, s1.cell_list ,'Failed',verbose=True )
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list ,'Failed',verbose=True )
        
        
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
    
    def test_3d_output_validation(self):
        particle_positions = np.array([[0.0, 0.5, 2.0], [1.5, 4.5, 6.0], [2.5, 2.5, 2.5], 
                                       [3.0, 4.5, 3.5], [4.5, 5.5, 7], [5.5, 3.5, 10],
                                       [6.0, 5.0, 11], [6.5, 9.5, 10.5], [7.0, 1.1, 0.2], 
                                       [8.25, 2.0, 1.0], [9.75, 9.0, 10.5], [10.0 ,8.0, 10.5 ] ])
        box_space = np.array([10.5 , 11.0, 10.5 ])
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
        
        
