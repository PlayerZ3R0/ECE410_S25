import numpy as np
import cupy as cp  # Import CuPy
import gym
import random
import time
import matplotlib.pyplot as plt

def train_gpu(episodes, learning_rate, gamma, epsilon, decay_rate):
    """
    Trains the Q-Learning agent using GPU acceleration with CuPy.
    """
    # Create the FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=True)
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # Initialize Q-table on the GPU using CuPy
    qtable = cp.zeros((state_space, action_space))

    # Convert scalar parameters to CuPy arrays for GPU operations
    lr_gpu = cp.asarray(learning_rate)
    gamma_gpu = cp.asarray(gamma)

    # Start timer
    start_time = time.time()

    # Training loop
    for episode in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Exploration-exploitation trade-off
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                # Exploit: use argmax on the GPU Q-table
                # cp.argmax returns a 0-d array, so we get the item
                action = int(cp.argmax(qtable[state, :]).item())

            # Take action and observe outcome
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Convert reward to a CuPy array for GPU calculation
            reward_gpu = cp.asarray(reward)

            # Q-table update rule performed entirely on the GPU
            qtable[state, action] = qtable[state, action] + lr_gpu * (
                reward_gpu + gamma_gpu * cp.max(qtable[new_state, :]) - qtable[state, action]
            )

            state = new_state
        
        # Decay epsilon
        epsilon = np.exp(-decay_rate * episode)

    # End timer
    end_time = time.time()
    
    # Return training time and the Q-table (moved back to CPU for analysis)
    return end_time - start_time, cp.asnumpy(qtable)


def train_cpu(episodes, learning_rate, gamma, epsilon, decay_rate):
    """
    The original pure Python/NumPy version for comparison.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)
    state_space = env.observation_space.n
    action_space = env.action_space.n
    
    qtable = np.zeros((state_space, action_space))
    
    start_time = time.time()

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
                
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
            )
            state = new_state
            
        epsilon = np.exp(-decay_rate * episode)
        
    end_time = time.time()
    return end_time - start_time, qtable

# --- Main Execution and Benchmarking ---
if __name__ == "__main__":
    # Hyperparameters
    episodes = 20000
    learning_rate = 0.8
    gamma = 0.95
    epsilon = 1.0
    decay_rate = 0.001

    print("--- Running CPU Benchmark ---")
    cpu_time, cpu_qtable = train_cpu(episodes, learning_rate, gamma, epsilon, decay_rate)
    print(f"CPU Training Time: {cpu_time:.4f} seconds\n")

    print("--- Running GPU Benchmark ---")
    try:
        gpu_time, gpu_qtable = train_gpu(episodes, learning_rate, gamma, epsilon, decay_rate)
        print(f"GPU Training Time: {gpu_time:.4f} seconds\n")
        
        speedup = cpu_time / gpu_time
        print(f"--- Comparison ---")
        if speedup > 1:
            print(f"GPU version was {speedup:.2f}x faster.")
        else:
            slowdown = 1 / speedup
            print(f"GPU version was {slowdown:.2f}x *slower* than the CPU version.")

    except ImportError:
        print("CuPy is not installed. Skipping GPU benchmark.")
    except Exception as e:
        print(f"An error occurred during GPU execution: {e}")
        print("This might happen if you don't have a compatible NVIDIA GPU and CUDA toolkit.")

