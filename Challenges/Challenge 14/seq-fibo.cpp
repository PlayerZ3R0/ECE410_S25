// The Challenge of Parallelizing Fibonacci
// The core of the Fibonacci sequence is its recurrence relation: F(n) = F(n-1) + F(n-2). To calculate any number in the sequence, you must know the previous two numbers. This creates a dependency chain that is fundamentally hostile to parallelization. You cannot calculate F(20) at the same time as F(10) without one of them waiting for the other's prerequisites.

// A GPU achieves speed by having thousands of threads execute the same instruction on different data simultaneously. A direct parallelization of a single Fibonacci sequence calculation is therefore impossible.

// So, how can we approach this in CUDA? The "trick" is to change the problem slightly. Instead of having all threads collaborate to build one sequence, we will launch N threads and task each thread i with calculating the i-th Fibonacci number from scratch. This is a computationally redundant but parallelizable approach. Each thread will independently and sequentially calculate the sequence up to its assigned index.

// Sequential CPU Implementation
// First, let's establish a baseline with a simple, efficient iterative implementation in C++. This will serve as our benchmark.

#include <chrono>
#include <iostream>
#include <vector>

// Use 64-bit unsigned integers to avoid overflow for larger numbers
using fib_type = unsigned long long;

/*
 * A simple iterative function to compute the first N Fibonacci numbers
 * and store them in a vector.
 */
void fibonacci_cpu(int n, std::vector<fib_type>& fib_sequence) {
    if (n <= 0) return;

    fib_sequence.resize(n);

    if (n >= 1) fib_sequence[0] = 0;
    if (n >= 2) fib_sequence[1] = 1;

    for (int i = 2; i < n; ++i) {
        // This is the dependency: F(i) needs F(i-1) and F(i-2)
        fib_sequence[i] = fib_sequence[i - 1] + fib_sequence[i - 2];
    }
}

int main() {
    const int N = 1 << 20;  // 2^20 = 1,048,576

    std::vector<fib_type> sequence(N);

    std::cout << "--- Running Sequential CPU Implementation ---" << std::endl;
    std::cout << "Calculating the first " << N << " Fibonacci numbers..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    fibonacci_cpu(N, sequence);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_time - start_time;

    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms" << std.endl;

    // Note: The Fibonacci sequence grows very rapidly. Beyond F(93), the result
    // will overflow a 64-bit unsigned integer. We are computing them anyway
    // to measure performance, but the higher values will be incorrect due to overflow.
    // std::cout << "F(90) = " << sequence[90] << std::endl;
    // std::cout << "F(93) = " << sequence[93] << std::endl;

    return 0;
}
