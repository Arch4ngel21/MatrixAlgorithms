import numpy as np


# Function for dividing square matrix into 4 quarters
def quarters(X: np.ndarray):
    width, height = X.shape
    half_width, half_height = width // 2, height // 2

    return (
        X[:half_width, :half_height],
        X[:half_width, half_height:],
        X[half_width:, :half_height],
        X[half_width:, half_height:],
    )


def find_closest_multiple_of_2(value: int):
    res = 1
    while res < value:
        res = res << 1
    return res


def strassen(A: np.ndarray, B: np.ndarray):
    if A.size == 1:
        return A[0][0] * B
    elif B.size == 1:
        return B[0][0] * A

    # Fancy check for multiple of 2 in matrix shapes
    if (
        A.shape != B.shape
        or bin(A.shape[0]).count("1") != 1
        or bin(A.shape[1]).count("1") != 1
        or bin(B.shape[0]).count("1") != 1
        or bin(B.shape[1]).count("1") != 1
    ):
        # new shape values are multiples of 2, the closest to maximum values of A and B shapes,
        # but higher than them or equal
        new_shape = (
            find_closest_multiple_of_2(np.max([A.shape[0], B.shape[0]])),
            find_closest_multiple_of_2(np.max([A.shape[1], B.shape[1]])),
        )

        # casting A and B to matrices of 0 with a new dimensions
        A_zeros = np.zeros(shape=new_shape)
        B_zeros = np.zeros(shape=new_shape)
        A_zeros[: A.shape[0], : A.shape[1]] = A
        B_zeros[: B.shape[0], : B.shape[1]] = B
        A = A_zeros
        B = B_zeros

    # division to 4 quarters
    a, b, c, d = quarters(A)
    e, f, g, h = quarters(B)

    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    # calculating the 4 quarters of the result matrix
    quarter_lu = p5 + p4 - p2 + p6
    quarter_ru = p1 + p2
    quarter_ld = p3 + p4
    quarter_rd = p1 + p5 - p3 - p7

    return np.hstack(
        (np.vstack((quarter_lu, quarter_ld)), np.vstack((quarter_ru, quarter_rd)))
    )


if __name__ == "__main__":
    test_A = np.array(
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [2, 2, 2, 2],
         [1, 1, 1, 1]]
    )

    test_B = np.array(
        [[1, 1, 1, 1, 1],
         [2, 2, 2, 2, 1],
         [3, 3, 3, 3, 1],
         [2, 2, 2, 2, 1]]
    )

    test_A_2 = np.array(
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [2, 2, 2, 2]], dtype=np.float64
    )

    test_B_2 = np.array(
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [2, 2, 2, 2]], dtype=np.float64
    )

    print(strassen(test_A, test_B))
    print(strassen(test_A_2, test_B_2))
