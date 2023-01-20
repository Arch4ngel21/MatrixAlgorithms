from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.utils.extmath import randomized_svd

# from ordering import preprocess_matrix, plot_ordering

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


def compress_matrix(A, t_min, t_max, s_min, s_max, r, epsilon):
    U, D, V = randomized_svd(A[t_min:t_max, s_min:s_max], n_components=r)

    # r = len(D) - 1

    # print(len(D), D, '\n')
    if (D < epsilon).all():
        v = Node(
            0,
            (t_max - t_min) * (s_max - s_min),
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
        (t_max - t_min) * (s_max - s_min),
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
):
    n, m = A.shape

    assert 0 <= t_min <= t_max <= n
    assert 0 <= s_min <= s_max <= m

    U, D, V = randomized_svd(A[t_min:t_max, s_min:s_max], n_components=r + 1)

    if D.size <= r or D[r] < epsilon:
        v = compress_matrix(A, t_min, t_max, s_min, s_max, r, epsilon)
    else:
        v = Node(0, (t_max - t_min) * (s_max - s_min), [], t_min=t_min, t_max=t_max, s_min=s_min, s_max=s_max)

        t_new_max = t_min + (t_max - t_min) // 2
        s_new_max = s_min + (s_max - s_min) // 2

        # left-up
        v.sons.append(
            create_tree(A, t_min, t_new_max, s_min, s_new_max, r, epsilon)
        )
        # right-up
        v.sons.append(
            create_tree(A, t_min, t_new_max, s_new_max, s_max, r, epsilon)
        )
        # left-down
        v.sons.append(
            create_tree(A, t_new_max, t_max, s_min, s_new_max, r, epsilon)
        )
        # right-down
        v.sons.append(
            create_tree(A, t_new_max, t_max, s_new_max, s_max, r, epsilon)
        )

    return v


def matrix_vector_mult(A, B, d=None):
    if d is None:
        if isinstance(A, Node):
            K, L, M = A.t_max - A.t_min, len(B), len(B[0])
        elif isinstance(B, Node):
            K, L, M = len(A), B.t_max - B.t_min, B.s_max - B.s_min
        else:
            K, L, M = len(A), len(B), len(B[0])
    else:
        K, L, M = d

    if isinstance(A, Node):
        if len(A.sons) == 0:
            if A.rank == 0:
                return np.zeros(shape=(K, M))
            else:
                A = A.U @ A.V
                C = np.zeros(shape=(K, M))
                return C + A @ B

        else:
            A11, A12, A21, A22 = A.sons
            B11 = B[:L // 2, :M // 2]
            B12 = B[:L // 2, M // 2:]
            B21 = B[L // 2:, :M // 2]
            B22 = B[L // 2:, M // 2:]

    elif isinstance(B, Node):
        if len(B.sons) == 0:
            if B.rank == 0:
                return np.zeros(shape=(K, M))
            else:
                B = B.U @ B.V

                C = np.zeros(shape=(K, M))
                return C + A @ B
        else:
            A11 = A[:K // 2, :L // 2]
            A12 = A[:K // 2, L // 2:]
            A21 = A[K // 2:, :L // 2]
            A22 = A[K // 2:, L // 2:]
            B11, B12, B21, B22 = B.sons
    else:

        print("Error - 2 regular matrices!")
        raise ValueError

    C = np.zeros(shape=(K, M))

    A11B11 = matrix_vector_mult(A11, B11, (K // 2, L // 2, M // 2))
    A12B21 = matrix_vector_mult(A12, B21, (K // 2, L - L // 2, M // 2))

    A11B12 = matrix_vector_mult(A11, B12, (K // 2, L // 2, M - M // 2))
    A12B22 = matrix_vector_mult(A12, B22, (K // 2, L - L // 2, M - M // 2))

    A21B11 = matrix_vector_mult(A21, B11, (K - K // 2, L // 2, M // 2))
    A22B21 = matrix_vector_mult(A22, B21, (K - K // 2, L - L // 2, M // 2))

    A21B12 = matrix_vector_mult(A21, B12, (K - K // 2, L // 2, M - M // 2))
    A22B22 = matrix_vector_mult(A22, B22, (K - K // 2, L - L // 2, M - M // 2))

    C[:K // 2, :M // 2] += A11B11
    C[:K // 2, :M // 2] += A12B21
    C[:K // 2, M // 2:] += A11B12
    C[:K // 2, M // 2:] += A12B22
    C[K // 2:, :M // 2] += A21B11
    C[K // 2:, :M // 2] += A22B21
    C[K // 2:, M // 2:] += A21B12
    C[K // 2:, M // 2:] += A22B22

    return C


def matrix_vector_add(v: Node, A: np.ndarray):
    if len(v.sons) == 0:
        if v.rank > 0:
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
            U1 = np.concatenate((v.U, w.U), axis=1)
            V1 = np.concatenate((v.V, w.V), axis=0)
            return U1 @ V1
        elif v.rank == 0:
            return w.U @ w.V
        elif w.rank == 0:
            return v.U @ v.V

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

    raise ValueError("Error in matrix_matrix_add - unhandled case")


def matrix_matrix_mult(v: Node, w: Node):
    K = v.t_max - v.t_min
    M = w.s_max - w.s_min

    if len(v.sons) == 0 and len(w.sons) == 0:
        if v.rank == 0 and w.rank == 0:
            return np.zeros(shape=(K, M))
        elif v.rank != 0 and w.rank != 0:
            return v.U @ (v.V @ w.U) @ w.V
        elif v.rank == 0 or w.rank == 0:
            return np.zeros(shape=(K, M))
        else:
            print("Weird error xd")
            raise ValueError

    if len(v.sons) > 0 and len(w.sons) > 0:
        A11, A12, A21, A22 = v.sons
        B11, B12, B21, B22 = w.sons

        C = np.zeros(shape=(K, M))

        A11B11 = matrix_matrix_mult(A11, B11)
        A11B12 = matrix_matrix_mult(A11, B12)
        A12B21 = matrix_matrix_mult(A12, B21)
        A12B22 = matrix_matrix_mult(A12, B22)
        A21B11 = matrix_matrix_mult(A21, B11)
        A21B12 = matrix_matrix_mult(A21, B12)
        A22B21 = matrix_matrix_mult(A22, B21)
        A22B22 = matrix_matrix_mult(A22, B22)

        C[:K//2, :M//2] += A11B11
        C[:K//2, :M//2] += A12B21
        C[:K//2, M//2:] += A11B12
        C[:K//2, M//2:] += A12B22
        C[K//2:, :M//2] += A21B11
        C[K//2:, :M//2] += A22B21
        C[K//2:, M//2:] += A21B12
        C[K//2:, M//2:] += A22B22
        return C

    if len(v.sons) > 0 and len(w.sons) == 0:
        C = np.zeros(shape=(K, M))

        if w.rank != 0:
            U, V = w.U, w.V
            u_mid = U.shape[0] // 2
            v_mid = V.shape[1] // 2

            U1, U2 = U[:u_mid, :], U[u_mid:, :]
            V1, V2 = V[:, :v_mid], V[:, v_mid:]

            A11 = U1 @ V1
            A12 = U1 @ V2
            A21 = U2 @ V1
            A22 = U2 @ V2
            B11, B12, B21, B22 = v.sons

            B11A11 = matrix_vector_mult(B11, A11)
            B11A12 = matrix_vector_mult(B11, A12)
            B12A21 = matrix_vector_mult(B12, A21)
            B12A22 = matrix_vector_mult(B12, A22)
            B21A11 = matrix_vector_mult(B21, A11)
            B21A12 = matrix_vector_mult(B21, A12)
            B22A21 = matrix_vector_mult(B22, A21)
            B22A22 = matrix_vector_mult(B22, A22)

            C[:K//2, :M//2] += B11A11
            C[:K//2, :M//2] += B12A21
            C[:K//2, M//2:] += B11A12
            C[:K//2, M//2:] += B12A22
            C[K//2:, :M//2] += B21A11
            C[K//2:, :M//2] += B22A21
            C[K//2:, M//2:] += B21A12
            C[K//2:, M//2:] += B22A22
        return C

    if len(v.sons) == 0 and len(w.sons) > 0:
        C = np.zeros(shape=(K, M))

        if v.rank != 0:
            U, V = v.U, v.V
            u_mid = U.shape[0] // 2
            v_mid = V.shape[1] // 2

            U1, U2 = U[:u_mid, :], U[u_mid:, :]
            V1, V2 = V[:, :v_mid], V[:, v_mid:]

            A11 = U1 @ V1
            A12 = U1 @ V2
            A21 = U2 @ V1
            A22 = U2 @ V2
            B11, B12, B21, B22 = w.sons

            A11B11 = matrix_vector_mult(A11, B11)
            A11B12 = matrix_vector_mult(A11, B12)
            A12B21 = matrix_vector_mult(A12, B21)
            A12B22 = matrix_vector_mult(A12, B22)
            A21B11 = matrix_vector_mult(A21, B11)
            A21B12 = matrix_vector_mult(A21, B12)
            A22B21 = matrix_vector_mult(A22, B21)
            A22B22 = matrix_vector_mult(A22, B22)

            C[:K//2, :M//2] += A11B11
            C[:K//2, :M//2] += A12B21
            C[:K//2, M//2:] += A11B12
            C[:K//2, M//2:] += A12B22
            C[K//2:, :M//2] += A21B11
            C[K//2:, :M//2] += A22B21
            C[K//2:, M//2:] += A21B12
            C[K//2:, M//2:] += A22B22
        return C

    print("matrix_mul end error!!!!!!!!!!!!!!!!!!!")


class HierarchicalMatrix:
    def __init__(self,
                 M: np.ndarray,
                 r: int,
                 epsilon=1e-08,
                 preprocess=False):

        # if preprocess:
        #     print("Finding an ordering for matrix...", end="")
        #     M = preprocess_matrix(M)
        #     print("Done")

        self.M = M
        self.size = M.size
        self.shape = M.shape
        self.r = r
        self.epsilon = epsilon

        self.head = create_tree(self.M, 0, self.shape[0], 0, self.shape[1], self.r, self.epsilon)

    def __add__(self, other):
        assert self.shape == other.shape

        if isinstance(other, HierarchicalMatrix):
            return matrix_matrix_add(self.head, other.head)
        elif isinstance(other, np.ndarray):

            return matrix_vector_add(self.head, other)
        else:
            raise TypeError

    def __sub__(self, other):
        assert self.shape == other.shape

        other = -other

        if isinstance(other, HierarchicalMatrix):
            return matrix_matrix_add(self.head, other.head)
        elif isinstance(other, np.ndarray):
            return matrix_vector_add(self.head, other)
        else:
            raise TypeError

    def __matmul__(self, other):
        if isinstance(other, HierarchicalMatrix):
            return matrix_matrix_mult(self.head, other.head)
        elif isinstance(other, np.ndarray):
            return matrix_vector_mult(self.head, other)
        else:
            raise TypeError

    def __neg__(self):
        copy = deepcopy(self)
        copy._neg_matrix(copy.head)
        return copy

    def to_numpy(self):
        return self._compressed_matrix_deconstruction(self.head)

    def _neg_matrix(self, T: Node):
        if len(T.sons) == 0 and T.rank > 0:
            T.V = -T.V
            return

        for son in T.sons:
            self._neg_matrix(son)

    def _compressed_matrix_deconstruction(self, T: Node):
        if len(T.sons) == 0:
            if T.rank > 0:
                M = T.U @ T.V
                M[np.abs(M) < self.epsilon] = 0
                return M
            else:
                return np.zeros((T.t_max - T.t_min, T.s_max - T.s_min))

        else:
            return np.hstack(
                (
                    np.vstack(
                        (
                            self._compressed_matrix_deconstruction(T.sons[0]),
                            self._compressed_matrix_deconstruction(T.sons[2]),
                        )
                    ),
                    np.vstack(
                        (
                            self._compressed_matrix_deconstruction(T.sons[1]),
                            self._compressed_matrix_deconstruction(T.sons[3]),
                        )
                    ),
                )
            )


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

    D = gen_matrix(N, N, 0.9)
    E = gen_matrix(N, N, 0.9)

    compressed_D = HierarchicalMatrix(D, R, EPSILON)
    compressed_E = HierarchicalMatrix(E, R, EPSILON)

    actual_add_res = D + E
    hierarchical_matrix_add_res = compressed_D + compressed_E
    diff = np.sum(np.power(hierarchical_matrix_add_res - actual_add_res, 2))
    print("====================== Matrix Addition ======================")
    print('Squared error:', diff)

    actual_mul_res = D @ E
    hierarchical_matrix_mul_res = compressed_D @ compressed_E
    diff = np.sum(np.power(hierarchical_matrix_mul_res - actual_mul_res, 2))

    print("====================== Matrix Multiplication ======================")
    print('Squared error:', diff)

