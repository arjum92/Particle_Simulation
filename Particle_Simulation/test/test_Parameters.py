from Particle_Simulation.Parameters import Parameters
import unittest


class test_Parameters(unittest.TestCase):
    def test_kvector(self):
        Parameter = Parameters(temperature=1, box=[1, 1, 1], es_sigma=2, update_radius=0.5, cutoff_radius=1, K_cutoff=1,
                               particle_types=[0])
        print(Parameter.k_vector)
