from Particle_Simulation.System import System
from Particle_Simulation.MetropolisMonteCarlo import MetropolisMonteCarlo


class Simulation:
    
    def __init__(self, system, parameters):
        self.system = system
        self.parameters = parameters
        self.opt_system = None
        self.sim_systems = None

    def optimize(self, n_steps):
        raise NotImplementedError()

        current_system = System(self.system.particles, self.parameters)
        current_system.energy = calculate_energy() # not implemented yet

        for i in range(n_steps):
            trial_system = MetropolisMonteCarlo.generate_trial_configuration(current_system, self.parameters)
            trial_system.energy = calculate_energy() # not implemented yet
            current_system = MetropolisMonteCarlo.evaluate_trial_configuration_greedy(current_system, trial_system)

        self.opt_system = current_system

    def simulate(self, n_steps):
        raise NotImplementedError()

        current_system = System(self.system.particles, self.parameters)
        current_system.energy = calculate_energy() # not implemented yet
        self.sim_systems.append(current_system)

        for i in range(n_steps):
            trial_system = MetropolisMonteCarlo.generate_trial_configuration(self.sim_systems[-1], self.parameters)
            trial_system.energy = calculate_energy() # not implemented yet
            self.sim_systems.append(MetropolisMonteCarlo.evaluate_trial_configuration(self.sim_systems[-1], trial_system))

