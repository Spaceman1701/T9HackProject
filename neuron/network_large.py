import math
import random


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.nodes = []
        prev = None
        for size in sizes:
            layer = [Node(prev) for _ in range(size)]
            self.nodes.append(layer)
            prev = layer

    def feed_forward(self, inputs):
        for value, node in zip(inputs, self.nodes[0]):  # input layer
            node.value = value
        for layer in self.nodes:
            for node in layer:
                node.eval()
        return [n.value for n in self.nodes[-1]]

    def clear_network(self):
        for node in self.all_nodes():
            node.value = None
            node.error = None

    def __str__(self):
        output = "Network"
        for layer in self.nodes:
            output += '\n'
            for node in layer:
                output += '['
                for e in node.ins:
                    output += str(e) + ', '
                output += '], '

        return output

    def all_nodes(self):
        for layer in self.nodes:
            for node in layer:
                yield node

    def back_propigate(self, inputs, outputs, learning_rate):
        self.feed_forward(inputs)
        assert len(self.nodes[-1]) == len(outputs)
        for layer in reversed(self.nodes):
            for node, value in zip(layer, outputs):
                node.calc_error(value[0])
        for node in self.nodes[1]:
            node.update_weights(learning_rate)

        self.clear_network()

    def train(self, training_data, epochs, batch_sides, learning_rate):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batch = training_data[0: batch_sides]
            for inputs, outputs in batch:
                self.back_propigate(inputs, outputs, learning_rate)
            print("Epoch {0} finished.", j)


class Edge:
    def __init__(self, origin, target, weight):
        self.weight = weight
        self.origin = origin
        self.target = target

    def __str__(self):
        return str(self.weight)


class Node:
    def __init__(self, prev):
        self.outs = []
        self.ins = []
        self.bias = random.uniform(0.0, 1.0)
        self.value = None
        self.error = None
        if prev:
            for node in prev:
                e = Edge(node, self, random.uniform(0.0, 1.0))
                node.outs.append(e)
                self.ins.append(e)

    def eval(self):
        activation = sum([weight * input_value for weight, input_value in zip_edge_value(self.ins)]) + self.bias
        activation = 1.0 / (1.0 + math.exp(-activation))
        self.value = activation
        return activation

    def calc_error(self, expected):
        if self.error:
            return self.error
        if not self.outs:
            self.error = expected - self.value
        else:
            self.error = sum([e.weight * e.target.calc_error(expected) for e in self.outs])
        return self.error

    def update_weights(self, learning_rate):
        for e in self.ins:
            assert self.value is not None
            assert self.error is not None
            assert e.origin.value is not None
            e.weight += (learning_rate * d_eval_func(self.value) * self.error * e.origin.value)
        for out in self.outs:
            out.target.update_weights(learning_rate)



def zip_edge_value(inputs):
    for edge in inputs:
        yield edge.weight, edge.origin.value


def d_eval_func(output):
    return output * (1 - output)


if __name__ == '__main__':
    in_data = [[1, 1, 1]] * 1000
    out_data = [[[0.4]]] * 1000
    data = []
    for v_i, v_o in zip(in_data, out_data):
        data.append((v_i, v_o))

    n = Network([3, 2, 1])
    print(n.feed_forward([1, 1, 1]))
    n.train(data, 100, 200, 0.9)
    print(n.feed_forward([1, 1, 1]))