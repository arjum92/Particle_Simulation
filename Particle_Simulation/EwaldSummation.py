import numpy as np
import math


class EwaldSummation:
    VACUUM_PERMITTIVITY = 1

    @staticmethod
    def calculate_longranged_energy(system, parameters):
        position = system.neighbourlist.particle_positions
        particlenumber = len(position)
        longrange_energy = 1

        for k in parameters.k_vector:
            s_k = 0
            kn = np.linalg.norm(k)

            for i in range(particlenumber):
                particle_paramenter = parameters.particle_types[system.particles[i].type_index]
                s_k += particle_paramenter.charge * np.cos(np.dot(k, position[i]))
            longrange_energy += (s_k ** 2) * (np.e ** (((parameters.es_sigma ** 2) * (kn ** 2)) / 2)) / (kn ** 2)
        longrange_energy *= (np.prod(parameters.box) * EwaldSummation.VACUUM_PERMITTIVITY)

        return longrange_energy

    @staticmethod
    def calculate_selfinteraction_potential(particle, parameters):
        return parameters.particle_types[particle.type_index].charge ** 2

    @staticmethod
    def calculate_shortranged_potential(particle_1, particle_2, parameters):

        charge_1 = parameters.particle_types[particle_1.type_index].charge
        charge_2 = parameters.particle_types[particle_2.type_index].charge
        particle_distance = np.linalg.norm(particle_1.position -  particle_2.position)

        short_ranged_potential = ((charge_1 * charge_2) / (particle_distance)) * math.erfc((particle_distance) / (np.sqrt(2) * parameters.es_sigma))

        return short_ranged_potential

    @staticmethod
    def calculate_wrapped_shortranged_potential(particle_1, particle_2, distance, parameters):

        charge_1 = parameters.particle_types[particle_1.type_index].charge
        charge_2 = parameters.particle_types[particle_2.type_index].charge
        particle_distance = distance

        short_ranged_potential = ((charge_1 * charge_2) / (particle_distance)) * math.erfc((particle_distance) / (np.sqrt(2) * parameters.es_sigma))

        return short_ranged_potential
