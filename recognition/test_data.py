from loader.data_loader import load_data_wrapper
import math


data = load_data_wrapper()

for inputs, outputs in data[0]:
    print(outputs.index(max(outputs)))
