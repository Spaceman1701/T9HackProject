import math
import random


class Layer:
    def __init__(self, prev_size, size):
        self.size = size
        self.weights = [[random.uniform(0.0, 1.0) for _ in range(prev_size)] for _ in range(size)]
        self.biases = [random.uniform(0.0, 1.0) for _ in range(size)]

    def evaluate(self, net_input):
        results = list()
        for node in range(self.size):
            weights = self.weights[node]
            results.append(sigmoid(weights, self.biases[node], net_input))
        return results

    def nodes(self):
        for node_weights, bias in zip(self.weights, self.biases):
            yield node_weights, bias


def sigmoid(weights, bias, net_inputs):
    sig_sum = sum([weight * net_input - bias for weight, net_input in zip(weights, net_inputs)]);
    return 1.0 / (1.0 + math.exp(-sig_sum))


def sigmoid_derivative(weights, bias, net_inputs):
    return sigmoid(weights, bias, net_inputs) * (1 - sigmoid(weights, bias, net_inputs))