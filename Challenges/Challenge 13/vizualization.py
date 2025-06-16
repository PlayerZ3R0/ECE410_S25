# To visualize the results, we use Python's Matplotlib. The following script takes the output from the C++ program and generates a bar plot. This plot clearly shows the relationship between total time and kernel-only time as the problem size grows




import matplotlib.pyplot as plt
import numpy as np

# --- Data from the CUDA Benchmark ---
# This data is representative of what you would get by running the C++ code.
# Actual results would go here
problem_sizes_str = [
    "2^15", "2^16", "2^17", "2^18", "2^19", "2^20", 
    "2^21", "2^22", "2^23", "2^24", "2^25"
]
problem_sizes_n = [1 << i for i in range(15, 26)]

total_time_ms = [
    0.186, 0.231, 0.365, 0.650, 1.28, 2.45, 
    4.98, 9.85, 20.12, 41.50, 85.34
]
kernel_time_ms = [
    0.007, 0.011, 0.019, 0.035, 0.068, 0.132, 
    0.261, 0.520, 1.05, 2.18, 4.41
]

x = np.arange(len(problem_sizes_str))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, total_time_ms, width, label='Total Time (incl. Memory Transfer)', color='skyblue')
rects2 = ax.bar(x + width/2, kernel_time_ms, width, label='Kernel-Only Time', color='royalblue')

# --- Add labels, title, and custom x-axis tick labels, etc. ---
ax.set_ylabel('Execution Time (ms) - Logarithmic Scale')
ax.set_title('SAXPY Execution Time vs. Vector Size on GPU', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(problem_sizes_str, rotation=45, ha="right")
ax.set_yscale('log') # Use a logarithmic scale to see the smaller values clearly
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)

fig.tight_layout()
plt.show()
