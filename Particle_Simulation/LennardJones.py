import numpy as np

class LennardJones:

    '''
    TODO: Writing the LennardJones.Energy.Calculating
    '''
    
    @staticmethod
    def calculate_lennardjones_energy(system, parameters):
        raise NotImplementedError

        # couldnt test this yet

        #lj_energy = 0
        #for i in range(0,len(system.neighborlist.head)):
        #    j = system.neighborlist.list[system.neighborlist.head[i]]
        #    while(x != -1):
        #        lj_energy += LennardJones._calculate_lennardjones_potential(system.particles[i], system.particles[j])
        #        j = system.neighborlist.list[j]

        #return lj_energy


    @staticmethod
    def _calculate_lennardjones_potential(particle_1, particle_2, parameters):

        particle_distance = LennardJones._calculate_distance(particle_1, particle_2)
        attractive_term = (parameters.lj_sigma / particle_distance) ** 6
        repulsive_term = attractive_term ** 2
        lj_potential = 4 * parameters.lj_epsilon * (repulsive_term - attractive_term)

        return lj_potential

    # this function should be moved to a more general spot since it is needed for
    # the computation of the short-ranged Coulomb interaction energy as well and
    # is not lj exclusive
    @staticmethod
    def _calculate_distance(particle_1, particle_2):

        if len(particle_1.position) == len(particle_2.position):
            dim = len(particle_1.position)
        else:
            raise ValueError('The dimensions of the entered particles are not equal.')

        sum = 0
        for i in range(0,dim):
            sum += (particle_1.position[i] - particle_2.position[i]) ** 2
        distance = np.sqrt(sum)

        return distance
