class Parameters:
    def __init__(self):
        self.temperature = None
        self.es_sigma = None
        self.mc_update_radius = None
        self.particle_types = None

    def __init__(self, temperature, box, es_sigma, update_radius, particle_types, cutoff_radius):
        self.temperature = temperature
        self.box = box
        self.es_sigma = es_sigma
        self.update_radius = update_radius
        self.particle_types = particle_types
        self.cutoff_radius = cutoff_radius
