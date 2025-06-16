import numpy as np

class NeuralNetwork:
    """
    Implements a multi-layer feed-forward neural network with one hidden layer.
    It is trained using the backpropagation algorithm.
    """
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate=0.1):
        # Set random seed for reproducibility of results
        np.random.seed(42)

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.lr = learning_rate

        # Initialize weights and biases
        # Weights from Input layer to Hidden layer
        self.weights_hidden = np.random.uniform(size=(self.num_inputs, self.num_hidden))
        self.bias_hidden = np.random.uniform(size=(1, self.num_hidden))

        # Weights from Hidden layer to Output layer
        self.weights_output = np.random.uniform(size=(self.num_hidden, self.num_outputs))
        self.bias_output = np.random.uniform(size=(1, self.num_outputs))

    def _sigmoid(self, x):
        """The sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        # Note: x is the sigmoid output, not the raw sum
        return x * (1 - x)

    def _feedforward(self, inputs):
        """
        Performs the feedforward pass through the network.
        Returns the activations of both the hidden and output layers.
        """
        # Calculate hidden layer activation
        hidden_sum = np.dot(inputs, self.weights_hidden) + self.bias_hidden
        hidden_activation = self._sigmoid(hidden_sum)

        # Calculate output layer activation
        output_sum = np.dot(hidden_activation, self.weights_output) + self.bias_output
        output_activation = self._sigmoid(output_sum)

        return hidden_activation, output_activation

    def train(self, training_inputs, training_outputs, epochs):
        """
        Trains the neural network using backpropagation.
        """
        print("--- Starting Training for XOR Gate ---")
        for epoch in range(epochs):
            # Perform a feedforward pass for the entire training set
            hidden_layer_activation, output_layer_activation = self._feedforward(training_inputs)

            # --- Backward Pass (Backpropagation) ---

            # 1. Calculate Error at the Output Layer
            output_error = training_outputs - output_layer_activation
            
            # 2. Calculate the Output Layer Delta
            # This is the "gradient" for the output layer
            output_delta = output_error * self._sigmoid_derivative(output_layer_activation)

            # 3. Calculate Error at the Hidden Layer
            # Propagate the error back to the hidden layer
            hidden_error = output_delta.dot(self.weights_output.T)
            
            # 4. Calculate the Hidden Layer Delta
            hidden_delta = hidden_error * self._sigmoid_derivative(hidden_layer_activation)

            # --- Update Weights and Biases ---

            # 5. Update Output Layer Weights and Bias
            self.weights_output += hidden_layer_activation.T.dot(output_delta) * self.lr
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.lr

            # 6. Update Hidden Layer Weights and Bias
            self.weights_hidden += training_inputs.T.dot(hidden_delta) * self.lr
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr

            # Print the error at certain intervals
            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(training_outputs - output_layer_activation))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        print("--- Training Complete ---\n")
    
    def predict(self, inputs):
        """Makes a prediction for a given set of inputs."""
        _, prediction = self._feedforward(inputs)
        return prediction


# --- Prepare Data for XOR Gate ---
# Input data for XOR
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Expected output for XOR
xor_outputs = np.array([[0], [1], [1], [0]])

# --- Create and Train the Neural Network ---
# Network architecture: 2 input neurons, 2 hidden neurons, 1 output neuron
nn = NeuralNetwork(num_inputs=2, num_hidden=2, num_outputs=1, learning_rate=0.5)
nn.train(xor_inputs, xor_outputs, epochs=10000)

# --- Test the Trained Network ---
print("--- XOR Test Results ---")
for inputs, expected in zip(xor_inputs, xor_outputs):
    prediction = nn.predict(inputs)
    # Reshape inputs for printing
    inputs_reshaped = inputs.reshape(1, 2)
    print(f"Input: {inputs_reshaped[0]} -> Prediction: {prediction[0][0]:.4f} (Expected: {expected[0]})")
