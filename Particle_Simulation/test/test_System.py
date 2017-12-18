import numpy as np
import unittest
import numpy.testing as npt
from Particle_Simulation.System import System

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
        particle_positions = np.array([[1, 1], [0, 3], [3, 4], [5, 5], [10, 10], [10, 0]])
        box_space = np.array([10, 10])
        cutoff = 4.6
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
        
    def test_1d_NeighListlength(self):
        #Neighbour_list_length_consistency_1D
        particle_positions = np.array([[0.0], [1.5], [2.5], 
                                       [3.0], [4.5] , [5.5],
                                       [6], [6.5], [7], 
                                       [8.25],[9.75],[10] ])
        box_space = np.array([10])
        cutoff = 0.5
        no_particles = len(particle_positions)
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        NeighList_length =len(s1.particle_neighbour_list)
        self.assertEqual(no_particles, NeighList_length , 'Fail: NeighList length not equals no_of_particles') 
        
    def test_1d_CellListlength(self):
        #cell_list_length_consistency_1D
        particle_positions = np.array([[0.0], [1.5], [2.5], 
                                       [3.0], [4.5] , [5.5],
                                       [6] ])
        box_space = np.array([10])
        cutoff = 4
        refrence_cell_no = 2 
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        cell_list_length =len(s1.cell_list)
        self.assertEqual(refrence_cell_no, cell_list_length , 'Fail: CellList length not equals reference') 
    
    def test_2d_NeighListlength(self):
        #Neighbour_list_length_consistency_2D
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,9.0], [10.0 ,8.0 ] ])
        box_space = np.array([10.5 , 11])
        cutoff = 3.5
        no_particles = len(particle_positions)
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        NeighList_length =len(s1.particle_neighbour_list)
        self.assertEqual(no_particles, NeighList_length , 'Fail: NeighList length not equals no_of_particles') 
        
    def test_2d_CellListlength(self):
        #cell_list_length_consistency_1D
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0], [10.0 ,8.0 ] ])
        box_space = np.array([10.5 , 11.])
        cutoff = 3.5
        refrence_cell_no = 9 
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        cell_list_length =len(s1.cell_list)
        self.assertEqual(refrence_cell_no, cell_list_length , 'Fail: CellList length not equals reference') 
    
    def test_3d_NeighListlength(self):
        #Neighbour_list_length_consistency_3D
        particle_positions = np.array([[0.0, 0.5, 2.0], [1.5, 4.5, 6.0], [2.5, 2.5, 2.5], 
                                       [3.0, 4.5, 3.5], [4.5, 5.5, 7], [5.5, 3.5, 10],
                                       [6.0, 5.0, 11], [6.5, 9.5, 10.5], [7.0, 1.1, 0.2], 
                                       [8.25, 2.0, 1.0], [9.75, 9.0, 10.5], [10.0 ,8.0, 10.5 ] ])
        box_space = np.array([10.5 , 11, 10.5])
        cutoff = 3.5
        no_particles = len(particle_positions)
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        NeighList_length =len(s1.particle_neighbour_list)
        self.assertEqual(no_particles, NeighList_length , 'Fail: NeighList length not equals no_of_particles') 
        
    def test_3d_CellListlength(self):
        #cell_list_length_consistency_3D
        particle_positions = np.array([[0.0, 0.5, 2.0], [1.5, 4.5, 6.0], [2.5, 2.5, 2.5], 
                                       [3.0, 4.5, 3.5], [4.5, 5.5, 7], [5.5, 3.5, 10],
                                       [6.0, 5.0, 11], [6.5, 9.5, 10.5], [7.0, 1.1, 0.2], 
                                       [8.25, 2.0, 1.0], [9.75, 9.0, 10.5], [10.0 ,8.0, 10.5 ] ])
        box_space = np.array([10.5 , 11.0, 10.5 ])
        cutoff = 3.5
        refrence_cell_no = 27 
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        cell_list_length =len(s1.cell_list)
        self.assertEqual(refrence_cell_no, cell_list_length , 'Fail: CellList length not equals reference') 
    
    def test_dimension_mismatch(self):
        #particle_position_dimension_diff_with_box
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0], [10.0 ,8.0 ] ])
        box_space = np.array([10.5 , 11.0, 2.0])
        self.assertEqual(len(particle_positions), len(box_space), 'Fail: dim not equal')
        cutoff = 3.5
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
    
    def test_outofbound(self):
        #particle_position_out_of_box
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,12.0], [10.0 ,8.0 ] ])
        box_space = np.array([10.5 , 11.0 ])
        cutoff = 3.5
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist() 
    
    def test_particle_dim_mismatch(self):
        #particle_internal_dim_mismatch    
        particle_positions = np.array([[1],[2,0], [0], [3], [4], [0]])
        box_space = np.array([4])
        cutoff = 2
        s1 = System(particle_positions, box_space, cutoff)
        s1.construct_neighborlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)        
        
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
        particle_positions = np.array([[0.0, 0.5], [1.5, 4.5], [2.5, 2.5], 
                                       [3.0, 4.5], [4.5, 5.5], [5.5, 3.5],
                                       [6.0, 5.0], [6.5, 9.5], [7.0, 1.1], 
                                       [8.25, 2.0], [9.75,11.0], [10.0 ,8.0 ] ])
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
