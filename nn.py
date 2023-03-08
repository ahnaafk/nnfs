class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        #initialize weights and biases
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #calculate output values from inputs * weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
