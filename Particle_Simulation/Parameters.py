class Parameters:

    def __init__(self):
        self.temperature = None
        self.es_sigma = None
        self.mc_update_radius = None
        self.particle_types = None

    def __init__(self, temperature, es_sigma, mc_update_radius, particle_types):
        self.temperature = temperature
        self.es_sigma = es_sigma
        self.mc_update_radius = mc_update_radius
        self.particle_types = particle_types
