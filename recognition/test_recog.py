from loader.data_loader import load_data_wrapper
from neuron.network_large import Network


def get_value(output):
    max_index = 0
    for i in range(len(output)):
        if (output[i] > output[max_index]):
            max_index = i
    return i

def get_sucess(net, data):
    test_data = data[1]
    iter = 100
    correct = 0
    tested = iter
    for inputs, outputs in test_data:
        result = net.feed_forward(inputs)
        # print(result)
        result_value = get_value(result)
        if outputs == result_value:
            correct += 1
        if iter < 1:
            break
        iter -= 1

    print("finished")
    print(correct / tested)


data = load_data_wrapper()

net = Network([784, 20, 10])

get_sucess(net, data)

i = 25000
training = data[0]
for test_in, test_out in training:
    # print(test_in)
    # print(test_out)
    net.back_propigate(test_in, test_out, 0.9)
    i -= 1
    if i % 200 == 0:
        print(i)
    if i <= 0:
        break

get_sucess(net, data)
