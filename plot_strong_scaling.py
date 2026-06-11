#!/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

runtime = np.array([14.89, 7.51, 4.00, 2.16, 1.19, 1.13])
num_threads = 2**np.arange(6)

speedup = runtime[0] / runtime

plt.figure(figsize=(8, 5))
plt.loglog(num_threads, speedup, '-o', linewidth=2, label='Measured speedup')
plt.loglog(num_threads, num_threads, '--', linewidth=2, label='Ideal speedup')

plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.title('Strong scaling')

# Legend and grid
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()
plt.savefig("strong_scaling_ruche.pdf")
