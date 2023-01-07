from dataclasses import dataclass
from typing import List, Optional, Tuple
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

# ============================ Lab 3 ============================


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

        # left-up
        v.sons.append(
            create_tree(A, t_min, t_new_max, s_min, s_new_max, r, epsilon, tab + 1)
        )
        # right-up
        v.sons.append(
            create_tree(A, t_min, t_new_max, s_new_max, s_max, r, epsilon, tab + 1)
        )
        # left-down
        v.sons.append(
            create_tree(A, t_new_max, t_max, s_min, s_new_max, r, epsilon, tab + 1)
        )
        # right-down
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

# ===============================================================

# ============================ Lab 4 ============================


def matrix_vector_mult(v: Node, A: np.ndarray):
    if len(v.sons) == 0:
        if v.rank == 0:
            return np.zeros(A.shape[0])
        return v.U @ (v.V @ A)

    rows = A.shape[0]
    A_1 = A[:rows//2, :]
    A_2 = A[rows//2: rows, :]
    B_1_1 = matrix_vector_mult(v.sons[0], A_1)
    B_1_2 = matrix_vector_mult(v.sons[1], A_2)
    B_2_1 = matrix_vector_mult(v.sons[2], A_1)
    B_2_2 = matrix_vector_mult(v.sons[3], A_2)

    return np.vstack((
        B_1_1 + B_1_2, B_2_1 + B_2_2
    ))


def matrix_vector_add(v: Node, A: np.ndarray):
    if len(v.sons) == 0:
        if v.rank > 0:
            # U1 = U + v.U
            # V1 = V + v.V
            return A + (v.U @ v.V)
        else:
            return A
    else:
        mid_rows = A.shape[0] // 2
        mid_cols = A.shape[1] // 2

        A1, A2, A3, A4 = A[:mid_rows, :mid_cols], A[mid_rows:, :mid_cols], A[:mid_rows, mid_cols:], A[mid_rows:, mid_cols:]

        return np.hstack((
                np.vstack((matrix_vector_add(v.sons[0], A1), matrix_vector_add(v.sons[2], A2))),
                np.vstack((matrix_vector_add(v.sons[1], A3), matrix_vector_add(v.sons[3], A4)))
        ))


def matrix_matrix_add(v: Node, w: Node):
    if len(v.sons) == 0 and len(w.sons) == 0:
        if v.rank == 0 and w.rank == 0:
            return np.zeros(shape=(v.U.shape[0], v.V.shape[1]))
        elif v.rank != 0 and w.rank != 0:
            # U1 = v.U + w.U
            # V1 = v.V + w.V
            return v.U @ v.V + w.U @ w.V

    elif len(v.sons) > 0 and len(w.sons) > 0:
        return np.hstack((
            np.vstack((
                matrix_matrix_add(v.sons[0], w.sons[0]),
                matrix_matrix_add(v.sons[2], w.sons[2])
            )),
            np.vstack((
                matrix_matrix_add(v.sons[1], w.sons[1]),
                matrix_matrix_add(v.sons[3], w.sons[3])
            ))
        ))

    if len(v.sons) > 0 and len(w.sons) == 0:
        v, w = w, v

    if len(v.sons) == 0 and len(w.sons) > 0:
        U, V = v.U, v.V
        u_mid = U.shape[0] // 2
        v_mid = V.shape[1] // 2

        U1, U2 = U[:u_mid, :], U[u_mid:, :]
        V1, V2 = V[:, :v_mid], V[:, v_mid:]

        B1, B2, B3, B4 = w.sons

        return np.hstack((
            np.vstack((
                matrix_vector_add(B1, U1 @ V1),
                matrix_vector_add(B3, U2 @ V1)
            )),
            np.vstack((
                matrix_vector_add(B2, U1 @ V2),
                matrix_vector_add(B4, U2 @ V2)
            ))
        ))


def matrix_matrix_mult(v: Node, w: Node):
    if len(v.sons) == 0 and len(w.sons) == 0:
        if v.rank == 0 and w.rank == 0:
            return np.zeros(shape=(v.U.shape[0], v.V.shape[1]))
        elif v.rank != 0 and w.rank != 0:
            return v.U @ (v.V @ w.U) @ w.V

    if len(v.sons) > 0 and len(w.sons) > 0:
        A1, A2, A3, A4 = v.sons
        B1, B2, B3, B4 = w.sons

        return np.hstack((
            np.vstack((
                matrix_matrix_mult(A1, B1) + matrix_matrix_mult(A2, B3),
                matrix_matrix_mult(A3, B1) + matrix_matrix_mult(A4, B3)
            )),
            np.vstack((
                matrix_matrix_mult(A1, B2) + matrix_matrix_mult(A2, B4),
                matrix_matrix_mult(A3, B2) + matrix_matrix_mult(A4, B4)
            ))
        ))

    if len(v.sons) > 0 and len(w.sons) == 0:
        v, w = w, v

    if len(v.sons) == 0 and len(w.sons) > 0:
        U, V = v.U, v.V
        u_mid = U.shape[0] // 2
        v_mid = V.shape[1] // 2

        U1, U2 = U[:u_mid, :], U[u_mid:, :]
        V1, V2 = V[:, :v_mid], V[:, v_mid:]

        B1, B2, B3, B4 = w.sons

        return np.hstack((
            np.vstack((
                matrix_vector_mult(B1, U1 @ V1) + matrix_vector_mult(B3, U1 @ V2),
                matrix_vector_mult(B1, U2 @ V1) + matrix_vector_mult(B3, U2 @ V2)
            )),
            np.vstack((
                matrix_vector_mult(B2, U1 @ V1) + matrix_vector_mult(B4, U1 @ V2),
                matrix_vector_mult(B2, U2 @ V1) + matrix_vector_mult(B4, U2 @ V2)
            ))
        ))


# ===============================================================


def gen_matrix(n: int, m: int, zero_threshold: float):

    assert 0.0 <= zero_threshold <= 1.0

    A = np.random.uniform(0.0, 1.0, (n, m))

    zero_mask = np.random.uniform(0.0, 1.0, (n, m))

    A[zero_mask < zero_threshold] = 0

    return A


if __name__ == '__main__':
    EPSILON = 1e-08
    R = 5
    np.set_printoptions(precision=4)

    A = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ])

    B = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ])

    C = np.array([1, 2, 3, 4])

    N = 32

    D = gen_matrix(N, N, 0.1)
    E = gen_matrix(N, N, 0.1)

    compressed_D = create_tree(D, 0, N, 0, N, R, EPSILON)
    compressed_E = create_tree(E, 0, N, 0, N, R, EPSILON)

    actual_add_res = D + E
    hierarchical_matrix_add_res = matrix_matrix_add(compressed_D, compressed_E)
    diff = np.sum(np.power(hierarchical_matrix_add_res - actual_add_res, 2))
    print("====================== Matrix Addition ======================")
    print('Squared error:', diff)

    actual_mul_res = D @ E
    hierarchical_matrix_mul_res = matrix_matrix_mult(compressed_D, compressed_E)
    diff = np.sum(np.power(hierarchical_matrix_mul_res - actual_mul_res, 2))

    print("====================== Matrix Multiplication ======================")
    print('Squared error:', diff)
