import math
import random


class Layer:
    def __init__(self, prev_size, size):
        self.size = size
        self.weights = [[random.uniform(0.0, 1.0) for _ in range(prev_size)] for _ in range(size)]
        self.biases = [[random.uniform(0.0, 1.0) for _ in range(prev_size)] for _ in range(size)]

    def evaluate(self, net_input):
        results = list()
        for node in range(self.size):
            weights = self.weights[node]
            biases = self.biases[node]
            results.append(self.sigmoid(weights, biases, net_input))
        return results

    def sigmoid(self, weights, biases, net_inputs):
        sig_sum = weights * net_inputs - biases;
        return 1.0 / (1.0 + math.exp(-sig_sum))
