# Here is a Python program that simulates the systolic array for Bubble Sort. It includes a SystolicProcessor class to represent each PE and a SystolicArray class to manage the whole process. It also benchmarks the execution time for various input sizes


import time
import random
import matplotlib.pyplot as plt
import numpy as np

class SystolicProcessor:
    """A single Processing Element (PE) in the systolic array."""
    def __init__(self):
        self.value = None

    def step(self, input_val):
        """
        In one clock cycle, compare the input value with the stored value
        and pass the smaller one along.
        The larger value is kept. This simulates the compare-and-swap.
        """
        if self.value is None or input_val < self.value:
            # Swap
            temp = self.value
            self.value = input_val
            return temp
        else:
            return input_val

class SystolicArraySorter:
    """Simulates a 1D systolic array for sorting."""
    def __init__(self, size):
        self.size = size
        # Create N processing elements
        self.processors = [SystolicProcessor() for _ in range(size)]

    def sort(self, data):
        """
        Sorts the data by feeding it through the systolic array.
        The simulation takes 2*N cycles.
        """
        if len(data) != self.size:
            raise ValueError("Input data size must match the array size.")
        
        # We need 2*N cycles for the data to pass through and sort
        num_cycles = 2 * self.size
        
        # Use a copy of the data to not modify the original
        input_stream = list(data)
        
        for cycle in range(num_cycles):
            # The value entering the first processor in this cycle
            # For the first N cycles, it's the data. After that, it's "infinity".
            if input_stream:
                next_val = input_stream.pop(0)
            else:
                # Feed in "infinity" to push the remaining values through
                next_val = float('inf')

            # Propagate the value through the array of PEs
            for pe in self.processors:
                next_val = pe.step(next_val)
        
        # After 2*N cycles, the sorted values are stored in the processors
        return [pe.value for pe in self.processors]

def run_benchmark():
    """Runs sorting for various sizes and collects execution times."""
    sizes = [10, 100, 1000, 5000, 10000]
    times = []

    print("--- Running Systolic Array Sort Benchmark ---")
    print(f"{'Input Size':<15}{'Execution Time (s)':<20}")
    print("-" * 35)

    for size in sizes:
        # Create an array of the specified size with random numbers
        data_to_sort = [random.randint(0, size * 10) for _ in range(size)]
        
        sorter = SystolicArraySorter(size)

        start_time = time.perf_counter()
        
        # Perform the sort
        sorted_data = sorter.sort(data_to_sort)
        
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        times.append(duration)

        print(f"{size:<15}{duration:<20.6f}")

        # Verification step (optional)
        assert sorted_data == sorted(data_to_sort)

    return sizes, times

def visualize_results(sizes, times):
    """Visualizes the execution times using a bar plot."""
    x_pos = np.arange(len(sizes))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_pos, times, align='center', alpha=0.7, color='c')
    
    plt.xticks(x_pos, sizes)
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Array Size (N)')
    plt.title('Systolic Array Sort Performance')
    plt.yscale('log') # Use log scale as times grow quickly
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.show()

if __name__ == '__main__':
    benchmark_sizes, benchmark_times = run_benchmark()
    visualize_results(benchmark_sizes, benchmark_times)

