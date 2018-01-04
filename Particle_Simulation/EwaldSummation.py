import numpy as np


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
        raise NotImplementedError

    @staticmethod
    def _calculate_selfinteraction_energy(system, parameters):
        summation = 0
        prefactor = 1 / (2 * EwaldSummation.VACUUM_PERMITTIVITY * parameters.es_sigma * (2 * np.pi) ** (3 / 2))

        for i in range(0, len(system.particles)):
            summation += parameters.particle_types[system.particles[i].type_index].charge ** 2
        selfinteraction_energy = prefactor * summation

        return selfinteraction_energy
