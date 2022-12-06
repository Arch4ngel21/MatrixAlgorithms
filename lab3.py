from dataclasses import dataclass
from typing import List, Optional

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

    if (A[t_min:t_max, s_min:s_max] == 0).any():
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
        U=U[:, 1:rank],
        V=np.diag(D[1:rank]) @ V[1:rank, :],
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

    print("\t" * tab + f"{(t_min, t_max, s_min, s_max)}")
    U, D, V = randomized_svd(A[t_min:t_max, s_min:s_max], n_components=r)

    if (D < epsilon).all() or D.size == 1:
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
        print("\t" * tab + f"| {(T.t_min, T.t_max, T.s_min, T.s_max)}")
    else:
        for v in T.sons:
            _print_tree(v, ax, tab + 1)


def print_tree(T: Node):
    fig, ax = plt.subplots()
    ax.plot([0, T.t_max, T.t_max, 0, 0], [0, 0, T.s_max, T.s_max, 0], color="black")

    _print_tree(T, ax)

    plt.show()


if __name__ == "__main__":
    # A = np.array([
    #     [1, 3, 5],
    #     [6, 4, 8],
    #     [1, 1, 2]
    # ])

    B = np.random.uniform(0.0, 1.0, (32, 32))
    R = 5
    EPSILON = 1e-08

    t, s = B.shape

    tree = create_tree(B, 0, t, 0, s, R, EPSILON)
    print_tree(tree)
