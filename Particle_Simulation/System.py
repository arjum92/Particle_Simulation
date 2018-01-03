from Particle_Simulation.Energy import Energy


class System:
    def __init__(self, particles):
        self.particles = particles
        self.energy = Energy()

        # def __init__(self, particles, box, rc):
            # self.particles = particles
            # self.energy = Energy()
            # self.neighborlist = Neighborlist(particles, box, rc)

    def update_neighborlist(self):
        raise NotImplementedError
