import numpy as np
import random

class Layer:
    def __init__(self, prev_size, size):
        self.size = size
        self.weights = [[random.uniform(0.0, 1.0) for _ in range(size)] for _ in range(size)]
        self.bias = [[random.uniform(0.0, 1.0) for _ in range(size)] for _ in range(size)]


l = Layer(10, 10)

print(l.weights)