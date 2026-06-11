#!/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

runtime = np.array([15.70, 15.65, 16.2, 17.64, 22.33])
num_threads = 2**np.arange(5)

efficiency = runtime[0] / runtime

plt.figure(figsize=(8, 5))
plt.plot(num_threads, efficiency, '-o', linewidth=2, label='Measured efficiency')
plt.plot(num_threads, np.ones_like(num_threads), '--', linewidth=2, label='Ideal efficiency')

plt.xlabel('Number of threads')
plt.ylabel('Efficiency')
plt.title('Weak scaling')

# Legend and grid
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()
plt.savefig("weak_scaling_ruche.pdf")
