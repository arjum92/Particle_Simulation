from Particle_Simulation.EwaldSummation import EwaldSummation
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.ParticleType import ParticleType
from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
import numpy as np
import unittest


class test_EwaldSummation(unittest.TestCase):
    def test_longrange(self):
        particle_type = ParticleType(name="Natrium", mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=0, box=np.array([1, 1, 1]), es_sigma=0.5, update_radius=1,
                                particle_types=particle_type, cutoff_radius=0.5)
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
        particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=0)
        particles = np.array([particle_1, particle_2])
        system = System(particles, parameters)
        E = EwaldSummation._calculate_longranged_energy(self, system=system, parameters=parameters)
        print(E)
