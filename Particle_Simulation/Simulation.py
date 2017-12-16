class Simulation:

    def __init__(self, system, parameters):
        self.system = system
        self.parameters = parameters
        self.opt_system = None
        self.sim_systems = None

    def optimize(self, n_steps):
        raise NotImplementedError()

    def simulate(self):
        raise NotImplementedError()
