import numpy as np

dvalues = np.array([
[1., 1., 1.],
[2., 2., 2.],
[3., 3., 3.]
])

weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]).T

#derivative w.r.t inputs
dinputs = np.dot(dvalues, weights.T)

print(dinputs)

inputs = np.array([
    [1, 2, 3, 2.5],
    [2., 5., -1., 2],
    [-1.5, 2.7, 3.3, -0.8]
])

dweights = np.dot(inputs.T, dvalues)

print(dweights)

biases = np.array([
    [2, 3, 0.5]
])

dbiases = np.sum(dvalues, axis=0, keepdims=True)

print(dbiases)