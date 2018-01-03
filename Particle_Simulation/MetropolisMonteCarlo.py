import numpy as np
from Energy import Energy


class MetropolisMonteCarlo:

    BOLTZMANN_CONSTANT = 1

    @staticmethod
    def generate_trial_configuration(system, parameters):

        n_particles = len(system.particles)
        update_probability = np.random.rand(1)[0]
        
        for i in range(0, n_particles):
            random_number = np.random.rand(1)[0]
            if random_number <= update_probability:
                system.particles[i].position = MetropolisMonteCarlo._generate_trial_position(system.particles[i].position,
                                                                                             parameters)

        return system

    @staticmethod
    def evaluate_trial_configuration_greedy(system, trial_system):

        if trial_system.energy.overall_energy <= system.energy.overall_energy:
            return trial_system
        else:
            return system

    @staticmethod
    def evaluate_trial_configuration(system, trial_system, parameters):

        beta = 1 / (MetropolisMonteCarlo.BOLTZMANN_CONSTANT * parameters.temperature)
        acceptance_probability = np.exp(-beta * (trial_system.energy.overall_energy - system.energy.overall_energy))

        if acceptance_probability >= 1:
            return trial_system
        else:
            random_number = np.random.rand(1)[0]
            if random_number <= acceptance_probability:
                return trial_system
            else:
                return system

    @staticmethod
    def _generate_trial_position(position, parameters):

        dimension = len(position)
        u = np.random.rand(1)[0]
        random_numbers = np.zeros(dimension)
        for i in range(0, dimension):
            random_numbers[i] = np.random.normal()

        numerator = parameters.update_radius * u ** (1 / dimension)
        nominator = 0
        for i in range(0, dimension):
            nominator += random_numbers[i] ** 2
        nominator = np.sqrt(nominator)
        fraction = numerator / nominator

        new_position = list(position + (random_numbers * fraction))
        new_position = MetropolisMonteCarlo._shift_position(new_position, parameters)

        return new_position

    @staticmethod
    def _shift_position(position, parameters):

        for i in range(0, len(position)):
            if position[i] > parameters.box[i]:
                position[i] = position[i] - parameters.box[i]
            elif position[i] < 0:
                position[i] = position[i] + parameters.box[i]

        for i in range(0, len(position)):
            if position[i] > parameters.box[i] or position[i] < 0:
                position = MetropolisMonteCarlo._shift_position(position, parameters)

        return position
