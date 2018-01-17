import numpy as np


class LennardJones:

    @staticmethod
    def calculate_potential(particle_1, particle_2, parameters):

        sigma = LennardJones._determine_sigma(particle_1, particle_2, parameters)
        epsilon = LennardJones._determine_epsilon(particle_1, particle_2, parameters)

        particle_distance = np.linalg.norm(particle_1.position - particle_2.position)
        attractive_term = (sigma / particle_distance) ** 6
        repulsive_term = attractive_term ** 2
        lj_potential = 4 * epsilon * (repulsive_term - attractive_term)

        return lj_potential

    @staticmethod
    def calculate_wrapped_potential(particle_1, particle_2, distance, parameters):

        sigma = LennardJones._determine_sigma(particle_1, particle_2, parameters)
        epsilon = LennardJones._determine_epsilon(particle_1, particle_2, parameters)

        particle_distance = distance
        attractive_term = (sigma / particle_distance) ** 6
        repulsive_term = attractive_term ** 2
        lj_potential = 4 * epsilon * (repulsive_term - attractive_term)

        return lj_potential

    @staticmethod
    def _determine_sigma(particle_1, particle_2, parameters):

        if particle_1.type_index == particle_2.type_index:
            sigma = parameters.particle_types[particle_1.type_index].lj_sigma
        else:
            sigma = (parameters.particle_types[particle_1.type_index].lj_sigma +
                     parameters.particle_types[particle_2.type_index].lj_sigma) / 2

        return sigma

    @staticmethod
    def _determine_epsilon(particle_1, particle_2, parameters):

        if particle_1.type_index == particle_2.type_index:
            epsilon = parameters.particle_types[particle_1.type_index].lj_epsilon
        else:
            epsilon = np.sqrt(parameters.particle_types[particle_1.type_index].lj_epsilon *
                              parameters.particle_types[particle_2.type_index].lj_epsilon)

        return epsilon