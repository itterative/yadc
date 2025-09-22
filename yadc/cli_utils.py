from time import time

class Timer:
    def __init__(self):
        self.start_t: float = 0
        self.end_t: float = 0

    def __enter__(self):
        self.start_t = time()
        return self
    
    def __exit__(self, *args):
        self.end_t = time()

    @property
    def elapsed(self):
        if self.start_t == 0 or self.end_t == 0:
            return 0

        return self.end_t - self.start_t
