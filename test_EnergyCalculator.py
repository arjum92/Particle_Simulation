
from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
from Particle_Simulation.LennardJones import LennardJones
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.EwaldSummation import EwaldSummation
from Particle_Simulation.Energy import Energy

import unittest


class test_Energy_Calculator(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
class test_LennardJones(unittest.TestCase):
    '''
    def test_calculate_distance_1(self):
        reference_distance = 11.0
        particle_1 = Particle(position=np.array([1, 0, 0]), type_index=1)
        particle_2 = Particle(position=np.array([12, 0, 0]), type_index=1)
        distance = LennardJones._calculate_distance(particle_1, particle_2)
        npt.assert_equal(reference_distance, distance, 'Failed', verbose=True)

    def test_calculate_distance_2(self):
        reference_distance = 3.5
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=1)
        particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=1)
        distance = LennardJones._calculate_distance(particle_1, particle_2)
        npt.assert_equal(reference_distance, distance, 'Failed', verbose=True)
        
    def test_calculate_lennardjones_potential(self):
        reference_potential = -0.000042499
        print(reference_potential)
        particle_type = ParticleType(name="Natrium", mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=0, es_sigma=0, update_radius=1, particle_types=particle_type,
                                box=np.array([1, 1, 1]), cutoff_radius=1, K_cutoff= 2)
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
        particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=0)
        lg_value = LennardJones._calculate_potential(particle_1, particle_2, parameters)
        lg_value_rounded = lg_value.round(decimals=9)
        print(lg_value_rounded)
        npt.assert_equal(reference_potential, lg_value_rounded, 'Failed', verbose=True)

def test_longrange(self):
    particle_type = ParticleType(name="Natrium", mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
    particle_type = np.array([particle_type])
    parameters = Parameters(temperature=0, box=np.array([1, 1, 1]), es_sigma=0.5, update_radius=1,
                            particle_types=particle_type, cutoff_radius=0.5, K_cutoff=2)
    particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
    particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=0)
    particles = np.array([particle_1, particle_2])
    system = System(particles, parameters)
    E = EwaldSummation.calculate_longranged_energy(system=system, parameters=parameters)
    print(E)

def test_determine_sigma(self):
    particle_positions = np.array([[9, 6, 4], [4, 5, 6], [1, 3, 6]])
    cell_list = np.array([1, 1, 1])
    box = np.array([12, 10, 10])
    cutoff_radius = 3
    #s1 = Neighbourlist(particle_positions, box, cutoff_radius)
    #particle_neighbour_list = s1.particle_neighbour_list()
    #cell_neighbour_list = s1.cell_neighbours()
    self.particle_neighbour_list = np.zeros(1, dtype=np.int32)
    self.cell_neighbour_list = np.zeros((1, 1, 1), dtype=np.int32)
    es_sigma = 1
    k_vector = np.array([[9, 6, 4], [4, 5, 6], [1, 3, 6]])
    charges = np.array([2, 1, 1.5])
    lj_sigmas = np.array([1, 2.5, 2])
    lj_epsilons = np.array([1, 2, 1.5])
    VACUUM_PERMITTIVITY = 1
    t1 = EnergyCalculator(box, cutoff_radius, es_sigma, charges, lj_sigmas, lj_epsilons, k_vector)
    sigma1 = t1._determine_sigma(0, 1)
    reference_Sigma_1 = 1.5
    self.assertEqual(reference_Sigma_1, sigma_1, msg='Failed: Actual and Desired are not equal')

def test_determine_epsilon(self):
    particle_positions = np.array([[4, 9, 6], [6, 4, 5], [6, 1, 3]])
    cell_list = np.array([1, 1, 2])
    box = np.array([12, 10, 11])
    cutoff_radius = 4
    #s1 = Neighbourlist(particle_positions, box, cutoff_radius)
    #particle_neighbour_list = s1.particle_neighbour_list()
    #cell_neighbour_list = s1.cell_neighbours()
    self.particle_neighbour_list = np.zeros(1, dtype=np.int32)
    self.cell_neighbour_list = np.zeros((1, 1, 1), dtype=np.int32)
    epsilon = 1
    k_vector = np.array([[6, 9, 4], [5, 4, 6], [6, 3, 1]])
    charges = np.array([2, 1, 2])
    lj_sigmas = np.array([1, 3, 2])
    lj_epsilons = np.array([1, 2, 2])
    VACUUM_PERMITTIVITY = 1
    t1 = EnergyCalculator(box, cutoff_radius, epsilon, charges, lj_sigmas, lj_epsilons, k_vector)
    epsilon1 = t1._determine_sigma(0, 1)
    reference_epsilon1 = 2
    self.assertEqual(reference_epsilon_1, epsilon_1, msg='Failed: Actual and Desired are not equal')

def test_calculate_shortrange_potential(self):
    reference_potential = -0.000042499
    print(reference_potential)
    particle_type = ParticleType(name="Hydrogen", mass=3, charge=1, lj_epsilon=1.20, lj_sigma=0.5)
    particle_type = np.array([particle_type])
    parameters = Parameters(temperature=0, es_sigma=0, update_radius=1, particle_types=particle_type,
                            box=np.array([1, 1, 2]), cutoff_radius=1, K_cutoff= 2)
    particle_1 = Particle(position=np.array([1, 2, 2]), type_index=0)
    particle_2 = Particle(position=np.array([2, 3, 5]), type_index=0)
    shortrange_value = shortrange._calculate_potential(particle_1, particle_2, parameters)
    shortrange_value_rounded = shortrange_value.round(decimals=9)
    print(shortrange_value_rounded)
    npt.assert_equal(reference_potential, lg_value_rounded, 'Failed', verbose=True)
def test_shortrange
