from collections import deque
import numpy as np
import matplotlib.pyplot as plt


def cuthill_mckee(A: np.ndarray) -> np.ndarray:
    n = len(A)

    visited = [False for _ in range(n)]

    degrees = np.count_nonzero(A, axis=1)
    R = []

    indices = np.expand_dims(np.arange(0, n, 1, dtype=np.int32), axis=1)
    degrees = np.expand_dims(degrees, axis=1)
    degree_sorted_indices = np.concatenate((degrees, indices), axis=1)

    degree_sorted_indices = np.array(sorted(degree_sorted_indices, key=lambda x: x[0]))

    q = deque()

    for i_deg, i in degree_sorted_indices:
        if not visited[i]:
            q.append(i)
            visited[i] = True

            while q:
                v = q.popleft()

                R.append(v)

                for w_deg, w in degree_sorted_indices:
                    if not visited[w]:
                        if A[v][w] != 0:
                            visited[w] = True
                            q.append(w)

    print("Ordering:", R)
    return R


def swap_nodes(matrix: np.ndarray, r):
    res_matrix = matrix.copy()
    for i in range(len(matrix)):
        res_matrix[i] = matrix[r[i]]

    res_matrix_2 = matrix.copy()
    for i in range(len(matrix)):
        res_matrix_2[:, i] = res_matrix[:, r[i]]

    return res_matrix_2


def plot_ordering(matrix: np.ndarray, res_matrix: np.ndarray):
    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Input matrix")
    ax1.matshow(matrix)
    ax2.set_title("Result matrix")
    ax2.matshow(res_matrix)
    plt.show()


def gen_matrix(n: int, m: int, zero_threshold: float):
    assert 0.0 <= zero_threshold <= 1.0

    A = np.random.uniform(0.0, 1.0, (n, m))

    zero_mask = np.random.uniform(0.0, 1.0, (n, m))

    A[zero_mask < zero_threshold] = 0

    return A


def gen_matrix_0_1(n: int, m: int, zero_threshold: float):
    assert 0.0 <= zero_threshold <= 1.0

    A = np.random.uniform(0.0, 1.0, (n, m))

    zero_mask = np.random.uniform(0.0, 1.0, (n, m))

    A[zero_mask < zero_threshold] = 0.0
    A[zero_mask >= zero_threshold] = 1.0

    return A


if __name__ == "__main__":
    A1 = np.array([
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    ])

    A2 = np.array([
        [1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    ])

    A2 = gen_matrix_0_1(100, 100, 0.9)

    R = cuthill_mckee(A2)

    A2_res = swap_nodes(A2, R)

    plot_ordering(A2, A2_res)
