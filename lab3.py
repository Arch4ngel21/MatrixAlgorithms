from dataclasses import dataclass
from typing import List, Optional
import timeit

"""
Jak przechodzi poniżej r, to U, V = A, I
"""


import numpy as np
from sklearn.utils.extmath import randomized_svd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


COLORS = list(mcolors.CSS4_COLORS.values())


@dataclass
class Node:
    rank: int
    size: int
    sons: List
    singular_values: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    t_min: int = None
    t_max: int = None
    s_min: int = None
    s_max: int = None


def compress_matrix(A, t_min, t_max, s_min, s_max, r):
    U, D, V = randomized_svd(A[t_min:t_max, s_min:s_max], n_components=r)

    if (A[t_min:t_max, s_min:s_max] == 0).all():
        v = Node(
            0,
            (t_max - t_min + 1) * (s_max - s_min + 1),
            [],
            None,
            t_min=t_min,
            t_max=t_max,
            s_min=s_min,
            s_max=s_max,
        )
        return v

    ro = D
    rank = r

    v = Node(
        r,
        (t_max - t_min + 1) * (s_max - s_min + 1),
        [],
        ro,
        U=U[:, :rank],
        V=np.diag(D[:rank]) @ V[:rank, :],
        t_min=t_min,
        t_max=t_max,
        s_min=s_min,
        s_max=s_max,
    )
    return v


def create_tree(
    A: np.ndarray,
    t_min: int,
    t_max: int,
    s_min: int,
    s_max: int,
    r: int,
    epsilon: float,
    tab=0,
):
    n, m = A.shape

    assert 0 <= t_min <= t_max <= n
    assert 0 <= s_min <= s_max <= m

    U, D, V = randomized_svd(A[t_min:t_max, s_min:s_max], n_components=r + 1)

    if D.size <= r or D[r] < epsilon:
        v = compress_matrix(A, t_min, t_max, s_min, s_max, r)
    else:
        v = Node(0, 0, [], t_min=t_min, t_max=t_max, s_min=s_min, s_max=s_max)

        t_new_max = t_min + (t_max - t_min) // 2
        s_new_max = s_min + (s_max - s_min) // 2

        v.sons.append(
            create_tree(A, t_min, t_new_max, s_min, s_new_max, r, epsilon, tab + 1)
        )
        v.sons.append(
            create_tree(A, t_min, t_new_max, s_new_max, s_max, r, epsilon, tab + 1)
        )
        v.sons.append(
            create_tree(A, t_new_max, t_max, s_min, s_new_max, r, epsilon, tab + 1)
        )
        v.sons.append(
            create_tree(A, t_new_max, t_max, s_new_max, s_max, r, epsilon, tab + 1)
        )

    return v


def _print_tree(T: Node, ax, tab=0):
    if len(T.sons) == 0:
        ax.add_patch(
            Rectangle(
                (T.t_min, T.s_min),
                (T.t_max - T.t_min),
                (T.s_max - T.s_min),
                color=COLORS[np.random.randint(0, 148)],
            )
        )
    else:
        for v in T.sons:
            _print_tree(v, ax, tab + 1)


def print_tree(T: Node, A: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot([0, T.t_max, T.t_max, 0, 0], [0, 0, T.s_max, T.s_max, 0], color="black")

    _print_tree(T, ax)

    plt.title(f"SVD for matrix A ({A.shape[0]} x {A.shape[1]})")
    plt.show()


def gen_matrix(n: int, m: int, zero_threshold: float):

    assert 0.0 <= zero_threshold <= 1.0

    A = np.random.uniform(0.0, 1.0, (n, m))

    zero_mask = np.random.uniform(0.0, 1.0, (n, m))

    A[zero_mask < zero_threshold] = 0

    return A


def compressed_matrix_deconstruction(T: Node, epsilon):
    if len(T.sons) == 0:
        if T.rank > 0:
            M = T.U @ T.V
            M[np.abs(M) < epsilon] = 0
            return M
        else:
            return np.zeros((T.t_max - T.t_min, T.s_max - T.s_min))

    else:
        return np.hstack(
            (
                np.vstack(
                    (
                        compressed_matrix_deconstruction(T.sons[0], epsilon),
                        compressed_matrix_deconstruction(T.sons[1], epsilon),
                    )
                ),
                np.vstack(
                    (
                        compressed_matrix_deconstruction(T.sons[2], epsilon),
                        compressed_matrix_deconstruction(T.sons[3], epsilon),
                    )
                ),
            )
        )


def decompressed_test(T: Node, A: np.ndarray, epsilon):
    decomposed_A = compressed_matrix_deconstruction(T, epsilon)

    assert decomposed_A.shape == A.shape

    print(
        "\nSquare error between A and decomposed A:",
        np.power(np.sum(A - decomposed_A), 2),
    )


if __name__ == "__main__":
    zero_percentage = 0.90

    A = gen_matrix(256, 256, zero_percentage)
    print("======================= Matrix A =======================")
    print(f"Zero percentage: {zero_percentage}")
    print(A)

    R = 5
    EPSILON = 1e-08

    t, s = A.shape

    tree = create_tree(A, 0, t, 0, s, R, EPSILON)

    decompressed_test(tree, A, EPSILON)

    print_tree(tree, A)

