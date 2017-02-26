import pickle
import gzip
import random

import numpy as np


def vectorize(i):
    # Returns a 10-dim vector with 1.0 in the ith location. Converts digit into desired output
    vector = np.zeros((10, 1))
    vector[i] = 1.0
    return vector.tolist()


def load_data():
    # Loads data from .gz and un-pickles it.
    file = gzip.open('./data/mnist.pk1.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(file, encoding='latin1')
    file.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [x.tolist() for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [x.tolist() for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [x.tolist() for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    index = random.randint(0, 40000)
    for i in range(28):
        print('')
        for j in range(28):
            if training_inputs[index][i * 28 + j] == 0:
                print('-', end='')
            else:
                print('@', end='')
    return training_data, validation_data, test_data

load_data_wrapper()
