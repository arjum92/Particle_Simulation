import numpy as np
from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
import Neighbourlist


class LennardJones:

    @staticmethod
    def calculate_energy(system, parameters):

        lj_energy = 0
        neighbour_cell_number = 3 ** system.neighbourlist.dim

        for i in range(system.neighbourlist.total_cell_number):
            particle_index_1 = system.neighbourlist.cell_list[i]

            while particle_index_1 != -1:

                for k in range(neighbour_cell_number):
                    cell_index = System.cell_neighbour_list[k][i][0]
                    particle_index_2 = system.neighbourlist.cell_list[cell_index]

                    while particle_index_2 != -1:

                        particle_1 = system.particles[particle_index_1]
                        particle_2 = system.particles[particle_index_2]

                        if particle_index_1 != particle_index_2:
                            if System.cell_neighbour_list[k][i][1] == 0:
                                if particle_index_1 < particle_index_2:
                                    lj_energy += LennardJones._calculate_potential(particle_1, particle_2, parameters)

                            elif System.cell_neighbour_list[k][i][1] != 0:

                                box_shift = LennardJones._determine_box_shift(i, k, parameters)
                                particle_2 = Particle(type_index=particle_2.type_index,
                                                      position=particle_2.position + box_shift)

                                if LennardJones._calculate_distance(particle_1, particle_2) < parameters.cutoff_radius:
                                    lj_energy += LennardJones._calculate_potential(particle_1, particle_2, parameters)

                        particle_index_2 = system.neighbourlist.particle_neighbour_list[particle_index_2]
                particle_index_1 = system.neighbourlist.particle_neighbour_list[particle_index_1]

        return lj_energy

    @staticmethod
    def _determine_box_shift(cell_index, cell_neighbour_index, parameters):

        box_shift = np.zeros((len(parameters.box)))
        if System.cell_neighbour_list[cell_neighbour_index][cell_index][1] != 0:
            for i in range(len(parameters.box)):
                if Neighbourlist.cell_shift_list[i][cell_neighbour_index] == 1:
                    box_shift[i] = parameters.box[i]
                elif Neighbourlist.cell_shift_list[i][cell_neighbour_index] == -1:
                    box_shift[i] = -parameters.box[i]
                else:
                    continue

        return box_shift

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
