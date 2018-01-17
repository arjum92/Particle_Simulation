import numpy as np
from numba import jit
from numba import jitclass
from numba import float32, int8, int32, int16, int64

from Particle import Particle
from System import System
from LennardJones import LennardJones
from Parameters import Parameters
from EwaldSummation import EwaldSummation
from Energy import Energy


class EnergyCalculator:

    def __init__(self, parameters):

        self.parameters = parameters


    def calculate_overall_energy(self, system):

        overall_energy = Energy()

        short_ranged_energy = self.calculate_shortranged_energy(system)
        overall_energy.lj_energy = short_ranged_energy[0]
        overall_energy.es_shortranged_energy = short_ranged_energy[1]
        overall_energy.es_selfinteraction_energy = self.calculate_shortranged_energy(system)

        return overall_energy

    def calculate_shortranged_energy(self, system):

        lj_energy = 0
        short_ranged_energy = 0
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
                                    particle_distance = np.linalg.norm(particle_1.position - particle_2.position)
                                    if particle_distance < self.parameters.cutoff_radius:
                                        lj_energy += LennardJones.calculate_potential(particle_1, particle_2,
                                                                                      self.parameters)
                                        short_ranged_energy += EwaldSummation.calculate_shortranged_potential(particle_1,
                                                                                                              particle_2,
                                                                                                              self.parameters)

                            elif System.cell_neighbour_list[k][i][1] != 0:

                                box_shift = self._determine_box_shift(i, k)
                                particle_2 = Particle(type_index=particle_2.type_index,
                                                      position=particle_2.position + box_shift)

                                particle_distance = np.linalg.norm(particle_1.position - particle_2.position)
                                if particle_distance < self.parameters.cutoff_radius:
                                    lj_energy += LennardJones.calculate_potential(particle_1, particle_2,
                                                                                  self.parameters)
                                    short_ranged_energy += EwaldSummation.calculate_shortranged_potential(particle_1,
                                                                                                          particle_2,
                                                                                                          self.parameters)

                        particle_index_2 = system.neighbourlist.particle_neighbour_list[particle_index_2]
                particle_index_1 = system.neighbourlist.particle_neighbour_list[particle_index_1]

        short_ranged_energy *= 1 / (8 * np.pi * Parameters.VACUUM_PERMITTIVITY)
        return [lj_energy, short_ranged_energy]

    def _determine_box_shift(self, cell_index, cell_neighbour_index):

        box_shift = np.zeros((len(self.parameters.box)))
        if System.cell_neighbour_list[cell_neighbour_index][cell_index][1] != 0:
            for i in range(len(self.parameters.box)):
                if Parameters.cell_shift_list[i][cell_neighbour_index] == 1:
                    box_shift[i] = self.parameters.box[i]
                elif Parameters.cell_shift_list[i][cell_neighbour_index] == -1:
                    box_shift[i] = -self.parameters.box[i]
                else:
                    continue

        return box_shift

    def calculate_selfinteraction_energy(self, system):

        summation = 0
        prefactor = 1 / (2 * Parameters.VACUUM_PERMITTIVITY * self.parameters.es_sigma * (2 * np.pi) ** (3 / 2))

        for i in range(0, len(system.particles)):
            summation += EwaldSummation.calculate_selfinteraction_potential(system.particles[i], self.parameters)
        selfinteraction_energy = prefactor * summation

        return selfinteraction_energy

    def calculate_shortranged_energy_2(self, system):

        lj_energy = 0
        short_ranged_energy = 0

        for i in range(0, len(system.particles)):
            for j in range(i + 1, len(system.particles)):
                particle_distance = np.linalg.norm(self.wrap_distance(system.particles[i].position - system.particles[j].position))

                if particle_distance < self.parameters.cutoff_radius:
                    lj_energy += LennardJones.calculate_wrapped_potential(system.particles[i], system.particles[j],
                                                                          particle_distance, self.parameters)
                    short_ranged_energy += EwaldSummation.calculate_wrapped_shortranged_potential(system.particles[i],
                                                                                                  system.particles[j],
                                                                                                  particle_distance,
                                                                                                  self.parameters)

        short_ranged_energy *= 1 / (8 * np.pi * Parameters.VACUUM_PERMITTIVITY)
        return [lj_energy, short_ranged_energy]

    def wrap_distance(self,distance):

        for i in range(len(distance)):
            while distance[i] >= 0.5 * self.parameters.box[i]:
                distance[i] -= self.parameters.box[i]
            while distance[i] < -0.5 * self.parameters.box[i]:
                distance[i] += self.parameters.box[i]

        return distance