import numpy as np

softmax_outputs = np.array(
    [[0.7, 0.1, 0.2],
     [0.1, 0.5, 0.4],
     [0.02, 0.9, 0.08]]
)

class_targets = [0, 1, 1]

neg_log1 = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])

# if one hot encoded test array

softmax_outputs = np.array(
    [[0.7, 0.1, 0.2],
     [0.1, 0.5, 0.4],
     [0.02, 0.9, 0.08]]
)

class_targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
    ])

correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)
neg_log2 = -np.log(correct_confidences)
