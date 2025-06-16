import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """
    Implements a single neuron with a sigmoid activation function,
    trained using gradient descent.
    """
    def __init__(self, num_inputs=2, learning_rate=0.1):
        # Initialize weights randomly, +1 for the bias
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.lr = learning_rate

    def _sigmoid(self, x):
        """The sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        fx = self._sigmoid(x)
        return fx * (1 - fx)

    def predict(self, inputs):
        """Pass inputs through the neuron to get a prediction."""
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self._sigmoid(weighted_sum)

    def train(self, training_inputs, training_outputs, epochs):
        """
        Train the neuron using gradient descent.
        """
        print("--- Starting Training ---")
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_output in zip(training_inputs, training_outputs):
                # 1. Forward pass
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                prediction = self._sigmoid(weighted_sum)

                # 2. Calculate the error
                error = expected_output - prediction
                total_error += abs(error)

                # 3. Backward pass (calculate adjustments)
                adjustment = error * self._sigmoid_derivative(weighted_sum)

                # 4. Update weights and bias
                self.weights += self.lr * adjustment * inputs
                self.bias += self.lr * adjustment

            # Print error at certain intervals
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Error: {total_error[0]:.4f}")

        print("--- Training Complete ---")
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias[0]:.4f}\n")

def plot_decision_boundary(neuron, inputs, outputs, title):
    """
    Plots the data points and the neuron's decision boundary.
    """
    plt.figure()
    # Plot data points
    for i, point in enumerate(inputs):
        if outputs[i] == 0:
            plt.plot(point[0], point[1], 'ro', markersize=10) # Red for 0
        else:
            plt.plot(point[0], point[1], 'bo', markersize=10) # Blue for 1

    # Plot the decision boundary line
    # The line is where w1*x + w2*y + b = 0.5 (for sigmoid)
    # We can approximate this as w1*x + w2*y + b = 0
    # y = (-w1*x - b) / w2
    x_vals = np.array([np.min(inputs[:, 0]) - 0.5, np.max(inputs[:, 0]) + 0.5])
    w = neuron.weights
    b = neuron.bias
    if w[1] != 0:
        y_vals = (-w[0] * x_vals - b) / w[1]
        plt.plot(x_vals, y_vals, 'k--')

    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])
    plt.show()