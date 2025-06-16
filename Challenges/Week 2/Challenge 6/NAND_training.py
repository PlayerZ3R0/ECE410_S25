# --- NAND Gate ---
print("--- Training for NAND Gate ---")
# Training data
nand_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_outputs = np.array([1, 1, 1, 0])

# Create and train the neuron
nand_neuron = Neuron(num_inputs=2, learning_rate=0.5)
nand_neuron.train(nand_inputs, nand_outputs, epochs=10000)

# Test the trained neuron
print("--- NAND Test Results ---")
for inputs in nand_inputs:
    prediction = nand_neuron.predict(inputs)
    print(f"Input: {inputs} -> Prediction: {prediction[0]:.4f} (Expected: {nand_outputs[np.all(nand_inputs==inputs, axis=1)][0]})")

# Plot the result
plot_decision_boundary(nand_neuron, nand_inputs, nand_outputs, "NAND Gate Decision Boundary")