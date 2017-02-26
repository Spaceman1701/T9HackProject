import random

from loader.data_loader import load_data_wrapper
from neuron.network_large import Network


def get_value(output):
    max_index = 0
    for i in range(len(output)):
        if output[i] > output[max_index]:
            max_index = i
        if output[i] > 1.0:
            print("bad mapping: ", output[i])
    return i


def get_sucess(net, data):
    test_data = data[1]
    iter = 50
    correct = 0
    tested = iter
    for inputs, outputs in test_data:
        result = net.feed_forward(inputs)
        print(result)
        result_value = get_value(result)
        if outputs == result_value:
            correct += 1
        if iter < 1:
            break
        iter -= 1

    print("finished")
    print(correct / tested)


data = load_data_wrapper()

net = Network([784, 30, 10])

get_sucess(net, data)

training = data[0]

test_data = [(x, y) for x, y in training]
md = [test_data[0]] * 100
print(md[0][1])

net.train(md, 30, 10, 8)

print(net.feed_forward(test_data[0][1]))

get_sucess(net, md[0])
