import numpy as np
import numpy.testing as npt
import unittest
from Particle_Simulation.Particle import Particle
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.LennardJones import LennardJones



'''
class test_LennardJones(unittest.TestCase) :
    
    def test_calculate_distance_1(self):
        l = LennardJones()
        reference_distance = 11.0
        particle_1 = Particle( np.array([1 ,0, 0]) , 1.642, 1.7)
        particle_2 = Particle( np.array([12 ,0, 0]), 1.642, 1.7)
        distance = l._calculate_distance(particle_1, particle_2)
        npt.assert_equal(reference_distance, distance ,'Failed',verbose=True )
    
    def test_calculate_distance_2(self):
        l = LennardJones()
        reference_distance = 3.5
        particle_1 = Particle( np.array([1 ,2, 3]) , 1.642, 1.7)
        particle_2 = Particle( np.array([2 ,3.5, 6]), 1.642, 1.8)
        distance = l._calculate_distance(particle_1, particle_2)
        npt.assert_equal(reference_distance, distance ,'Failed',verbose=True )
        
    def test_calculate_lennardjones_potential(self):
        l = LennardJones()
        reference_potential = -0.000042499
        print(reference_potential)
        parameters = Parameters(20, 0.6, lj_sigma = 0.5 , lj_epsilon = 1.25) 
        particle_1 = Particle( np.array([1 ,2, 3]) , 1.642, 1.7)
        particle_2 = Particle( np.array([2 ,3.5, 6]), 1.642, 1.8)
        lg_value = l._calculate_lennardjones_potential(particle_1, particle_2, parameters)
        lg_value_rounded = lg_value.round(decimals = 9) 
        print(lg_value_rounded)
        npt.assert_equal(reference_potential, lg_value_rounded ,'Failed',verbose=True )
        
                
if __name__ == '__main__' :
    unittest.main()

'''