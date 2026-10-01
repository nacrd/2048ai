"""Immutable orientation indices shared by compiled CPU and CUDA scoring."""
import numpy as np


def _paths():
    paths = np.zeros((5, 8, 36), dtype=np.uint8)
    for n in range(2, 7):
        for orientation in range(8):
            for rank in range(n * n):
                r, c = divmod(rank, n)
                if r % 2:
                    c = n - 1 - c
                if orientation & 1:
                    r, c = c, r
                if orientation & 2:
                    r = n - 1 - r
                if orientation & 4:
                    c = n - 1 - c
                paths[n - 2, orientation, rank] = r * n + c
    paths.flags.writeable = False
    return paths


SNAKE_PATHS = _paths()


def cuda_paths():
    sizes = []
    for size in SNAKE_PATHS:
        sizes.append('{' + ','.join('{' + ','.join(str(int(i)) for i in path) + '}' for path in size) + '}')
    return '__device__ __constant__ unsigned char snake_paths[5][8][36]={' + ','.join(sizes) + '};\n'
