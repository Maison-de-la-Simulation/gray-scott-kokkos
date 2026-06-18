#!/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from json import loads

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser(description="Plot performance in point-updates/s vs size")

parser.add_argument("-n", "--niter", help="number of iterations", type=int, default=100)
parser.add_argument("file", help="performance data file", type=Path)

args = parser.parse_args()

niter = args.niter
data = loads(args.file.read_text())

for benchmark in data:
    runtime = np.array(benchmark)

runtime_skx = np.array([0.002307, 0.002384, 0.006999, 0.018525, 0.137146, 0.570825, 2.296378, 9.337727])
runtime_v100 = np.array([0.001703, 0.001152, 0.001521, 0.004234, 0.012876, 0.047868, 0.186844, 0.746483])
runtime_a100 = np.array([0.001135, 0.001166, 0.001417, 0.002209, 0.007434, 0.025851, 0.099067, 0.393040])
rows = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800])
columns = rows
size = rows * columns

perf_skx = niter * size / runtime_skx * 1E-9
perf_v100 = niter * size / runtime_v100 * 1E-9
perf_a100 = niter * size / runtime_a100 * 1E-9

plt.figure(figsize=(8, 5))
plt.plot(size, perf_skx, '-o', linewidth=2, label='Intel Cascade Lake (20 cores)')
plt.semilogx(size, perf_v100, '-o', linewidth=2, label='Nvidia V100')
plt.semilogx(size, perf_a100, '-o', linewidth=2, label='Nvidia A100')

plt.xlabel('Problem size')
plt.ylabel('Performance (Mpoint-updates/s')
plt.title('Performance of different architectures')

# Legend and grid
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()
plt.savefig("performance_ruche.pdf")
