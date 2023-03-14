# layer_outputs = [4.8, 1.21, 2.385]

# E = 2.71828182846

# exp_values = []
# for output in layer_outputs:
#     exp_values.append(E ** output)

# print(f"Exponetiated Values: {exp_values}")

# norm_base = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)

# print("Normalized exponentiated values:")
# print(norm_values)

# print(f"Sum of normalized values: {sum(norm_values)}")

import numpy as np

layer_outputs = [1, 2, 3]

exp_values = np.exp(layer_outputs)
print(exp_values)
probabilities = exp_values / np.sum(exp_values)
print(probabilities)


