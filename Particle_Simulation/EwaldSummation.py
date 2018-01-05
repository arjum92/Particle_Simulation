import numpy as np
import math
from Particle_Simulation.LennardJones import LennardJones
from Particle_Simulation.System import System
from Particle_Simulation.Particle import Particle


class EwaldSummation:
    VACUUM_PERMITTIVITY = 1

    @staticmethod
    def calculate_overall_energy(system, parameters):
        overall_energy = EwaldSummation._calculate_longranged_energy(system, parameters) + \
                         EwaldSummation._calculate_shortranged_energy(system, parameters) - \
                         EwaldSummation._calculate_selfinteraction_energy(system, parameters)

        return overall_energy

    @staticmethod
    def _calculate_longranged_energy(self, system, parameters):
        position = system.neighbourlist.particle_positions
        particlenumber = len(position)
        longrange_energy = 0

        for k in system.k_vectors:
            s_k = 0

            for i in range(particlenumber):
                particle_paramenter = parameters.particle_types[system.particles[i].type_index]
                s_k += particle_paramenter.charge * np.e ** (1j * np.dot(k, position[i]))
            longrange_energy += (s_k ** 2) * (np.e ** (((parameters.es_sigma ** 2) * (k ** 2)) / 2)) / (k ** 2)
        longrange_energy *= 1 / (2 * (parameters.box ** 2) * EwaldSummation.VACUUM_PERMITTIVITY)

        return longrange_energy

    @staticmethod
    def _calculate_shortranged_energy(system, parameters):
        short_range_energy = 0
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
                                    if LennardJones._calculate_distance(particle_1,
                                                                        particle_2) < parameters.cutoff_radius:
                                        short_range_energy += EwaldSummation._calculate_shortranged_potential(
                                            particle_1=particle_1, particle_2=particle_2, parameters=parameters)
                            elif System.cell_neighbour_list[k][i][1] != 0:
                                box_shift = LennardJones._determine_box_shift(i, k, parameters)
                                particle_2 = Particle(type_index=particle_2.type_index,
                                                      position=particle_2.position + box_shift)
                                if LennardJones._calculate_distance(particle_1, particle_2) < parameters.cutoff_radius:
                                    short_range_energy += EwaldSummation._calculate_shortranged_potential(
                                        particle_1=particle_1, particle_2=particle_2, parameters=parameters)

                        particle_index_2 = system.neighbourlist.particle_neighbour_list[particle_index_2]
                particle_index_1 = system.neighbourlist.particle_neighbour_list[particle_index_1]
        short_range_energy *= 1 / (8 * np.pi * EwaldSummation.VACUUM_PERMITTIVITY)

        return short_range_energy

    @staticmethod
    def _calculate_selfinteraction_energy(system, parameters):
        summation = 0
        prefactor = 1 / (2 * EwaldSummation.VACUUM_PERMITTIVITY * parameters.es_sigma * (2 * np.pi) ** (3 / 2))

        for i in range(0, len(system.particles)):
            summation += parameters.particle_types[system.particles[i].type_index].charge ** 2
        selfinteraction_energy = prefactor * summation

        return selfinteraction_energy

    @staticmethod
    def _calculate_shortranged_potential(particle_1, particle_2, parameters):
        charge_1 = parameters.particle_types[particle_1.type_index].charge
        charge_2 = parameters.particle_types[particle_2.type_index].charge
        ri = particle_1.position
        rj = particle_2.position
        # nL = 1
        dis = np.linalg.norm(ri - rj)
        shortranged_potential = ((charge_1 * charge_2) / (dis)) * math.erfc((dis) / (np.sqrt(2) * parameters.es_sigma))
        return shortranged_potential
