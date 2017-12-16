class System:

    def __init__(self, particles):
        self.particles = particles

    def update_neighborlist(self):
        raise NotImplementedError