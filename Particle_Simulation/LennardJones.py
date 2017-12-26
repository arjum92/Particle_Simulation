import numpy as np


class LennardJones:

    @staticmethod
    def calculate_energy(system, parameters):
        raise NotImplementedError

        # couldnt test this yet
        #
        #lj_energy = 0
        #for i in range(0,len(system.neighborlist.head)):
        #    j = system.neighborlist.list[system.neighborlist.head[i]]
        #    while(x != -1):
        #        lj_energy += LennardJones._calculate_lennardjones_potential(system.particles[i], system.particles[j])
        #        j = system.neighborlist.list[j]
        #
        #return lj_energy

    @staticmethod
    def _calculate_potential(particle_1, particle_2, parameters):

        sigma = LennardJones._determine_sigma(particle_1, particle_2, parameters)
        epsilon = LennardJones._determine_epsilon(particle_1, particle_2, parameters)

        particle_distance = LennardJones._calculate_distance(particle_1, particle_2)
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










    # this function should be moved to a more general spot since it is needed for
    # the computation of the short-ranged Coulomb interaction energy as well and
    # is not lj exclusive
    @staticmethod
    def _calculate_distance(particle_1, particle_2):

        if len(particle_1.position) == len(particle_2.position):
            dim = len(particle_1.position)
        else:
            raise ValueError('The dimensions of the entered particles are not equal.')

        summation = 0
        for i in range(0, dim):
            summation += (particle_1.position[i] - particle_2.position[i]) ** 2
        distance = np.sqrt(summation)

        return distance
