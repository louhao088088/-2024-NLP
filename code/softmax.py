import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))  
    return exp_z / np.sum(exp_z)

z = [2.0, 1.0, 0.1]
softmax_output = softmax(z)
print("Softmax Output:", softmax_output)
