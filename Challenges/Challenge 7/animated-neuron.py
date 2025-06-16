import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Neuron:
    """
    Implements a single neuron with a sigmoid activation function,
    trained using gradient descent. It also records the history of
    its parameters during training for visualization.
    """
    def __init__(self, num_inputs=2, learning_rate=0.1):
        # Initialize weights and bias randomly for reproducibility
        np.random.seed(42)
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.lr = learning_rate
        
        # History lists to store parameters for animation
        self.weights_history = []
        self.bias_history = []
        self.error_history = []

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
        Train the neuron using gradient descent and record the history.
        """
        print("--- Starting Training ---")
        for epoch in range(epochs):
            total_error = 0
            
            # Store current state for animation
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias.copy())
            
            for inputs, expected_output in zip(training_inputs, training_outputs):
                # 1. Forward pass
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                prediction = self._sigmoid(weighted_sum)

                # 2. Calculate the error
                error = expected_output - prediction
                total_error += abs(error[0])

                # 3. Backward pass (calculate adjustments)
                adjustment = error * self._sigmoid_derivative(weighted_sum)

                # 4. Update weights and bias
                self.weights += self.lr * adjustment * inputs
                self.bias += self.lr * adjustment
            
            self.error_history.append(total_error)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Error: {total_error:.4f}")

        print("--- Training Complete ---")
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias[0]:.4f}\n")


# --- Prepare Data for NAND Gate ---
nand_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_outputs = np.array([1, 1, 1, 0])

# --- Create and Train the Neuron ---
# We'll train for a smaller number of epochs to make the animation concise
nand_neuron = Neuron(num_inputs=2, learning_rate=0.5)
training_epochs = 200
nand_neuron.train(nand_inputs, nand_outputs, epochs=training_epochs)

# --- Create the Animation ---
fig, ax = plt.subplots()

def setup_plot():
    """Sets up the static elements of the plot."""
    ax.clear()
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.grid(True)
    
    # Plot data points
    for i, point in enumerate(nand_inputs):
        if nand_outputs[i] == 0:
            ax.plot(point[0], point[1], 'ro', markersize=10, label='Output 0' if i == 3 else "") # Red for 0
        else:
            ax.plot(point[0], point[1], 'bo', markersize=10, label='Output 1' if i == 0 else "") # Blue for 1
    ax.legend(loc='upper right')

# Initialize the plot and the line object that will be updated
setup_plot()
line, = ax.plot([], [], 'k--', lw=2) # The decision boundary line

def update(frame):
    """The function called for each frame of the animation."""
    # Get weights and bias from the history for the current frame
    w = nand_neuron.weights_history[frame]
    b = nand_neuron.bias_history[frame]
    error = nand_neuron.error_history[frame]
    
    # Define x values for the line
    x_vals = np.array([-0.5, 1.5])
    
    # Calculate corresponding y values for the decision boundary
    # w1*x + w2*y + b = 0  => y = (-w1*x - b) / w2
    if w[1] != 0:
        y_vals = (-w[0] * x_vals - b) / w[1]
        line.set_data(x_vals, y_vals.flatten())
    
    ax.set_title(f"NAND Gate Learning\nEpoch: {frame + 1}/{training_epochs}, Error: {error:.4f}")
    return line,

# Create the animation object
# We use blit=True for a smoother animation
ani = FuncAnimation(fig, update, frames=len(nand_neuron.weights_history),
                    init_func=setup_plot, blit=False, interval=50, repeat=True)

# To display in environments like Jupyter or to save, convert to HTML5 video
# from IPython.display import HTML
# html_video = ani.to_html5_video()
# HTML(html_video)

# If running as a script, you might want to save it or show it
# ani.save('neuron_learning.mp4', writer='ffmpeg')
plt.show() # This will show the animation in a window if run locally.

