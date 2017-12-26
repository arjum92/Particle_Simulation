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
    def _calculate_longranged_energy(system, parameters):
        raise NotImplementedError

    @staticmethod
    def _calculate_shortranged_energy(system, parameters):
        raise NotImplementedError

    @staticmethod
    def _calculate_selfinteraction_energy(system, parameters):
        summation = 0
        prefactor = 1 / (2 * EwaldSummation.VACUUM_PERMITTIVITY * parameters.es_sigma * (2 * np.pi) ** (3/2))

        for i in range(0, len(system.particles)):
            summation += parameters.particle_types[system.particles[i].type_index].charge ** 2
        selfinteraction_energy = prefactor * summation

        return selfinteraction_energy
