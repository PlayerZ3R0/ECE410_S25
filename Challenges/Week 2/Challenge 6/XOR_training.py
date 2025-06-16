# --- XOR Gate ---
print("\n\n--- Training for XOR Gate ---")
# Training data
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([0, 1, 1, 0])

# Create and train the neuron
xor_neuron = Neuron(num_inputs=2, learning_rate=0.5)
xor_neuron.train(xor_inputs, xor_outputs, epochs=10000)

# Test the trained neuron
print("--- XOR Test Results ---")
for inputs in xor_inputs:
    prediction = xor_neuron.predict(inputs)
    print(f"Input: {inputs} -> Prediction: {prediction[0]:.4f} (Expected: {xor_outputs[np.all(xor_inputs==inputs, axis=1)][0]})")

# Plot the result
plot_decision_boundary(xor_neuron, xor_inputs, xor_outputs, "XOR Gate Decision Boundary")