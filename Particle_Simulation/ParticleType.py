class ParticleType:

    def __init__(self, name, mass, charge, lj_epsilon, lj_sigma):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.lj_epsilon = lj_epsilon
        self.lj_sigma = lj_sigma