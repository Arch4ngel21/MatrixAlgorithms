
import numpy as np
from sklearn.utils.extmath import randomized_svd


if __name__ == '__main__':
    A = np.array([[1, 2, 3, 4],
                  [2, 3, 4, 5],
                  [4, 5, 6, 7],
                  [6, 7, 8, 9]])

    print(A[0:2, 1:3])
    exit(0)

    U, D, V = randomized_svd(A, n_components=3)
    D = np.diag(D)
    V = D @ V

    print(U.shape, V.shape)
    u_mid = U.shape[0] // 2
    v_mid = V.shape[1] // 2

    U1, U2 = U[:u_mid, :], U[u_mid:, :]
    V1, V2 = V[:, :v_mid], V[:, v_mid:]

    print(U, '\n', V, '\n\n')

    print(U1,'\n', U2, '\n', V1, '\n', V2)

    print(np.hstack((
        np.vstack((
            U1 @ V1,
            U2 @ V1
        )),
        np.vstack((
            U1 @ V2,
            U2 @ V2
        ))
    )))
