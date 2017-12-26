import numpy as np
import numpy.testing as npt
import unittest
from  Particle_Simulation.System import System


class test_SystemInputs(unittest.TestCase):

    
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
        
        
    
