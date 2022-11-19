import numpy as np

from lab1 import strassen


def quarters(X: np.ndarray):
    width, height = X.shape
    half_width, half_height = width // 2, height // 2

    return (
        X[:half_width, :half_height],
        X[:half_width, half_height:],
        X[half_width:, :half_height],
        X[half_width:, half_height:],
    )


def inverse(A: np.ndarray):
    if A.shape == (2, 2):
        return 1 / (A[0][0] * A[1][1] - A[0][1] * A[1][0]) \
               * np.array([[A[1][1], -A[0][1]],
                           [-A[1][0], A[0][0]]])

    A11, A12, A21, A22 = quarters(A)

    A11_inv = inverse(A11)

    S22 = A22 - strassen(strassen(A21, A11_inv), A12)

    S22_inv = inverse(S22)

    C = strassen(strassen(strassen(A12, S22_inv), A21), A11_inv)
    I = np.eye(C.shape[0])
    B11 = strassen(A11_inv, (I + C))

    B12 = strassen(strassen(-A11_inv, A12), S22_inv)

    B21 = strassen(strassen(-S22_inv, A21), A11_inv)

    B22 = S22_inv

    return np.hstack((
        np.vstack((B11, B21)), np.vstack((B12, B22))
    ))


def LU(A):
    if A.shape == (2, 2):
        # System of equations based on example on Wikipedia:
        # https://en.wikipedia.org/wiki/LU_decomposition
        # System in undetermined, so we put additional restriction on L, that
        # diagonal of L has only ones.
        l11 = 1
        l22 = 1

        l12 = 0
        u21 = 0

        u11 = A[0][0]
        u12 = A[0][1]
        l21 = A[1][0] / u11
        u22 = A[1][1] - l21 * u12

        return np.array([[l11, l12],
                         [l21, l22]]), \
               np.array([[u11, u12],
                         [u21, u22]])

    A11, A12, A21, A22 = quarters(A)

    L11, U11 = LU(A11)

    U11_inv = inverse(U11)

    L21 = strassen(A21, U11_inv)

    L11_inv = inverse(L11)

    U12 = strassen(L11_inv, A12)

    L22 = A22 - strassen(strassen(strassen(A21, U11_inv), L11_inv), A12)
    S = np.copy(L22)

    L_S, U_S = LU(S)

    U22 = U_S

    L22 = L_S

    L_zeros = np.zeros(shape=(L_S.shape[0], L11.shape[1]))
    U_zeros = np.zeros(shape=(U11.shape[0], U_S.shape[1]))

    return np.hstack((
        np.vstack((L11, L21)), np.vstack((L_zeros, L22))
    )), np.hstack((
        np.vstack((U11, U_zeros)), np.vstack((U12, U22))
    ))


def det(A: np.ndarray):
    L, U = LU(A)
    # L diagonal has only 1's
    return np.prod(U.diagonal())


if __name__ == '__main__':
    A_test = np.array([[1, 2, 3, 4],
                       [3, 8, 5, 9],
                       [0, 8, 3, 5],
                       [0, 7, 7, 3]], dtype=np.float64)
    B_test = np.array([[4, 7],
                       [2, 6]])

    print('=========================== Input matrix ============================')
    print(A_test)

    print('======================== Inverse of a matrix ========================')
    print('inv(A):\n', inverse(A_test))
    print("\nCheck with numpy.linalg.inv() function:")
    print(np.linalg.inv(A_test))
    print('========================= LU decomposition ==========================')
    lu_res = LU(A_test)
    print('== L ==\n', lu_res[0], '\n')
    print('== U ==\n', lu_res[1], '\n')
    print('Check if strassen(L, U) gives input matrix A:')
    print(strassen(lu_res[0], lu_res[1]))
    print('============================ Determinant ============================')
    print('det(A) =', det(A_test))
    print('\nCheck with numpy.linalg.det() function:')
    print(np.linalg.det(A_test))
