import numpy as np

#shape of (3,4)
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

#shape of (3,4)
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2.0, 3.0, 0.5]

#transpose weights to be of shape (4,3)
layer_outputs = np.dot(inputs, np.array(weights).T) + bias

print(layer_outputs)