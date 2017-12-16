class InputError(Exception):
        def __init__(self, message):
            #self.expression = expression
            self.message = message