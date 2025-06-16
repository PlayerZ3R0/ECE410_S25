#include <iostream>
#include <vector>
#include <chrono>

// Use 64-bit unsigned integers to avoid overflow for larger numbers
using fib_type = unsigned long long;

/*
 * CUDA Kernel to compute Fibonacci numbers.
 * Each thread calculates one number in the sequence independently.
 * This is parallel, but computationally redundant.
 */
__global__ void fibonacci_kernel(fib_type* d_sequence, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (idx == 0) {
            d_sequence[idx] = 0;
            return;
        }
        if (idx == 1) {
            d_sequence[idx] = 1;
            return;
        }

        // Each thread must compute its own sequence up to its index
        fib_type a = 0;
        fib_type b = 1;
        fib_type c = 0;
        for (int i = 2; i <= idx; ++i) {
            c = a + b;
            a = b;
            b = c;
        }
        d_sequence[idx] = c;
    }
}


int main() {
    const int N = 1 << 20; // 2^20 = 1,048,576
    size_t bytes = N * sizeof(fib_type);

    // Host vector
    std::vector<fib_type> h_sequence(N);

    // Device pointer
    fib_type* d_sequence;
    cudaMalloc(&d_sequence, bytes);

    // Setup CUDA execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "--- Running Parallel CUDA Implementation ---" << std::endl;
    std::cout << "Calculating the first " << N << " Fibonacci numbers..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    fibonacci_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_sequence, N);

    // Synchronize to wait for the kernel to finish
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_time - start_time;
    
    std::cout << "CUDA Kernel Execution Time: " << gpu_duration.count() << " ms" << std::endl;
    
    // Copy result back to host to verify if needed
    // cudaMemcpy(h_sequence.data(), d_sequence, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sequence);
    
    return 0;
}
