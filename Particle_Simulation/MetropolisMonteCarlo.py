import numpy as np


class MetropolisMonteCarlo:

    BOLTZMANN_CONSTANT = 1


    @staticmethod
    def generate_trial_configuration(system, update_radius):

        n_particles = len(system.particles)
        update_probability = np.random.rand(1)[0]
        for i in range(0, n_particles):
            if np.random.rand(1)[0] <= update_probability:
                system.particles[i].position = MetropolisMonteCarlo._generate_trial_position(system.particles[i].position,
                                                                                             update_radius)
        return system

    @staticmethod

    def evaluate_trial_configuration_greedy(system, trial_system):
        raise NotImplementedError

        if trial_system.energy <= system.energy:
            return trial_system
        else:
            return system

    @staticmethod
    def evaluate_trial_configuration(system, trial_system, parameters):
        raise NotImplementedError

        beta = 1 / (MetropolisMonteCarlo.BOLTZMANN_CONSTANT * parameters.temperature)

        acceptance_probability = np.exp(-beta * (trial_system.energy - system.energy))

        if acceptance_probability >= 1:
            return trial_system
        else:
            random_number = np.random.rand(1)[0]
            if random_number <= acceptance_probability:
                return trial_system
            else:
                return system


    @staticmethod
    def _generate_trial_position(position, update_radius):

        dimension = len(position)
        u = np.random.rand(1)[0]
        random_numbers = np.zeros(dimension)
        for i in range(0, dimension):
            random_numbers[i] = np.random.normal()

        numerator = update_radius * u ** (1 / dimension)
        nominator = 0
        for i in range(0, dimension):
            nominator += random_numbers[i] ** 2
        nominator = np.sqrt(nominator)
        fraction = numerator / nominator

        new_position = position + (random_numbers * fraction)

        return new_position
