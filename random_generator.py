import random
import pandas as pd
import numpy as np

# df = pd.read_csv('dataset/test.csv')

# np.savetxt(r'random_inputs.txt', df.text, fmt='%s')

def generate_text():
    inputs = []

    with open('dataset/random_inputs.txt', 'r') as file:
        return file.readlines()[random.randint(0, 9999)]