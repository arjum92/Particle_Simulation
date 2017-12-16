class Parameters:

    def __init__(self):
        self.temperature = None
        self.es_sigma = None
        self.lj_sigma = None
        self.lj_epsilon = None

    def __init__(self, temperature, es_sigma, lj_sigma, lj_epsilon):
        self.temperature = temperature
        self.es_sigma = es_sigma
        self.lj_sigma = lj_sigma
        self.lj_epsilon = lj_epsilon


