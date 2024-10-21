class Constraint:

    def __init__(self, function):
        self.function = function

    def __call__(self, x, *args, **kwargs):
        return self.function(x, *args, **kwargs)
