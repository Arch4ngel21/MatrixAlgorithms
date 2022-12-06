import numpy as np


if __name__ == '__main__':
    A = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7]
    ])

    zeros_mask = np.random.uniform(0.0, 1.0, A.shape)

    zeros_mask = zeros_mask < 0.5

    A[zeros_mask] = 0

    print(A)

