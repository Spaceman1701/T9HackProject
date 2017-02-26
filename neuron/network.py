from neuron.layer import Layer


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = [Layer(0, sizes[0])]
        for i, size in enumerate(sizes[1:]):
            self.layers.append(Layer(sizes[i], size))  # this probably works, don't touch it

    def __str__(self):
        output = "Network: \n"
        for l in self.layers:
            output += "Layer weights: " + str(l.weights) + '\n'
        return output

    def feed_foward(self, inputs):
        output = []
        for layer in self.layers:
            inputs = layer.evaluate(inputs)
            output.append(inputs.copy())
        return output


print(Network([2, 3, 2]))




