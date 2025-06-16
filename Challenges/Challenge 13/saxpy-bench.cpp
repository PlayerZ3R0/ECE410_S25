// The CUDA C++ Benchmarking Code

// This program is a modified version of the standard SAXPY example. It's designed to loop through various large problem sizes, from N = 2^15 to N = 2^25. For each size, it meticulously times two critical components using cudaEvent_t, as hinted in the challenge:

// Kernel Execution Time: The actual time the GPU spends running the saxpy computation.
// Total Execution Time: The end-to-end time, including allocating memory on the GPU, transferring data from CPU to GPU, kernel execution, and transferring the results back.




#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip> // For std::fixed and std::setprecision

// CUDA Kernel for SAXPY: Y = a*X + Y
__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main(void)
{
  // --- Benchmark Setup ---
  std::vector<long long> problem_sizes;
  for (int i = 15; i <= 25; ++i) {
      problem_sizes.push_back(1LL << i);
  }

  float a = 2.0f; // Scalar 'a'

  // --- CUDA Event Timers ---
  cudaEvent_t start_total, stop_total;
  cudaEvent_t start_kernel, stop_kernel;
  checkCudaError(cudaEventCreate(&start_total), "Failed to create start_total event");
  checkCudaError(cudaEventCreate(&stop_total), "Failed to create stop_total event");
  checkCudaError(cudaEventCreate(&start_kernel), "Failed to create start_kernel event");
  checkCudaError(cudaEventCreate(&stop_kernel), "Failed to create stop_kernel event");


  std::cout << std::left << std::setw(15) << "Vector Size"
            << std::left << std::setw(25) << "Total Time (ms)"
            << std::left << std::setw(25) << "Kernel Time (ms)" << std::endl;
  std::cout << std::string(65, '-') << std::endl;

  // --- Loop through problem sizes ---
  for (long long n : problem_sizes) {
    long long n_bytes = n * sizeof(float);

    // 1. Allocate host memory
    float *h_x = new float[n];
    float *h_y = new float[n];

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
      h_x[i] = 1.0f;
      h_y[i] = 2.0f;
    }

    // --- Time the full operation (mem alloc, copy, kernel, copy back) ---
    checkCudaError(cudaEventRecord(start_total), "Failed to record start_total event");

    // 2. Allocate device memory
    float *d_x, *d_y;
    checkCudaError(cudaMalloc(&d_x, n_bytes), "Failed to allocate d_x");
    checkCudaError(cudaMalloc(&d_y, n_bytes), "Failed to allocate d_y");

    // 3. Transfer data from host to device
    checkCudaError(cudaMemcpy(d_x, h_x, n_bytes, cudaMemcpyHostToDevice), "Failed to copy h_x to d_x");
    checkCudaError(cudaMemcpy(d_y, h_y, n_bytes, cudaMemcpyHostToDevice), "Failed to copy h_y to d_y");

    // --- Time just the kernel execution ---
    checkCudaError(cudaEventRecord(start_kernel), "Failed to record start_kernel event");

    // 4. Launch the SAXPY kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x, d_y);
    checkCudaError(cudaGetLastError(), "SAXPY kernel launch failed");

    checkCudaError(cudaEventRecord(stop_kernel), "Failed to record stop_kernel event");
    // --- End kernel timing ---

    // 5. Transfer result back from device to host
    checkCudaError(cudaMemcpy(h_y, d_y, n_bytes, cudaMemcpyDeviceToHost), "Failed to copy d_y to h_y");
    
    checkCudaError(cudaEventRecord(stop_total), "Failed to record stop_total event");
    // --- End total timing ---

    // Synchronize to make sure all operations are complete
    checkCudaError(cudaEventSynchronize(stop_total), "Failed to synchronize stop_total event");

    float total_time_ms = 0;
    float kernel_time_ms = 0;
    checkCudaError(cudaEventElapsedTime(&total_time_ms, start_total, stop_total), "Failed to get total elapsed time");
    checkCudaError(cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel), "Failed to get kernel elapsed time");

    std::cout << std::left << std::setw(15) << n
              << std::left << std::setw(25) << std::fixed << std::setprecision(6) << total_time_ms
              << std::left << std::setw(25) << std::fixed << std::setprecision(6) << kernel_time_ms << std::endl;

    // 6. Free device and host memory
    checkCudaError(cudaFree(d_x), "Failed to free d_x");
    checkCudaError(cudaFree(d_y), "Failed to free d_y");
    delete[] h_x;
    delete[] h_y;
  }
  
  // Clean up events
  checkCudaError(cudaEventDestroy(start_total), "Failed to destroy start_total event");
  checkCudaError(cudaEventDestroy(stop_total), "Failed to destroy stop_total event");
  checkCudaError(cudaEventDestroy(start_kernel), "Failed to destroy start_kernel event");
  checkCudaError(cudaEventDestroy(stop_kernel), "Failed to destroy stop_kernel event");

  return 0;
}
