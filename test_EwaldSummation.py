from Particle_Simulation.EwaldSummation import EwaldSummation
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.ParticleType import ParticleType
from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
import numpy as np
import unittest


class test_EwaldSummation(unittest.TestCase):
         def test_longrange(self):
             particle_type = ParticleType(name="Natrium",    mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
             particle_type = np.array([particle_type])
             parameters = Parameters(temperature=0, box=np.array([1, 1, 1]), es_sigma=0.5, update_radius=1,
                                particle_types=particle_type, cutoff_radius=0.5, K_cutoff=2)
             particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
             particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=0)
             particles = np.array([particle_1, particle_2])
             system = System(particles, parameters)
E=    EwaldSummation.calculate_longranged_energy(system=system, parameters=parameters)
             print(E)

          def test_shortrange_potential(self):
              reference_potential = -0.000042499  
              print(reference_potential)
              particle_type = ParticleType(name="Hydrogen", mass=3, charge=1, shortrange_epsilon=1.25, shortrange_sigma=0.5)
              particle_type = np.array([particle_type])
              parameters = Parameters(temperature=0, es_sigma=0, update_radius=1, particle_types=particle_type,
                        box=np.array([1, 2, 1]), cutoff_radius=1)
              particle_1 = Particle(position=np.array([1, 2, 2]), type_index=0)
              particle_2 = Particle(position=np.array([2, 3.5, 4]), type_index=0)
              shortrange_value = shortrange._calculate_potential(particle_1, particle_2, parameters)
              shortrange_value_rounded = shortrange_value.round(decimals=9)
              print(shortrange_value_rounded)
              npt.assert_equal(reference_potential, shortrange_value_rounded, 'Failed', verbose=True)

       def test_wrapped_shortrange_potential(self):
                reference_potential = -0.000042499
            print(reference_potential)
            particle_type = ParticleType(name="Natgerium", mass=1, charge=3, shortrange_epsilon=1.25, shortrange_sigma=0.5)
            particle_type = np.array([particle_type])
            parameters = Parameters(temperature=0, es_sigma=0, update_radius=1, particle_types=particle_type,
                    box=np.array([2, 1, 1]), cutoff_radius=1)
            particle_1 = Particle(position=np.array([1, 2, 2]), type_index=0)
            particle_2 = Particle(position=np.array([2, 3.5, 2]), type_index=0)
            shortrange_value = shortrange.       _calculate_potential(particle_1, particle_2, parameters)
    shortrange_value_rounded =          shortrange_value.round(decimals=9)
             print(shortrange_value_rounded)
             npt.assert_equal(reference_potential,        shortrange_value_rounded, 'Failed', verbose=True)
             distance = shortrange._calculate_distance(particle_1, particle_2)
            npt.assert_equal(reference_distance, distance, 'Failed', verbose=True)

           def test_selfinteraction_potential(self):
                particle_1 = Particle(position=np.array([1, 0, 0]), type_index=1)
                particle_2 = Particle(position=np.array([12, 0, 0]), type_index=1)
                parameters = Parameters(temperature=0, es_sigma=0, update_radius=1, particle_types=particle_type,
                            box=np.array([2, 1, 1]), cutoff_radius=1)
              npt.assert_equal(selfinteraction_potential, 'Failed', verbose=True)

