import numpy as np

def householder_solve(A, B):
    m, n = A.shape

    for k in range(n-1):
        v = A[k:, k].copy()
        v[0] = v[0] - np.sign(v[0]) * np.linalg.norm(v)
        v = v / np.linalg.norm(v)

        A[k:, k:] = A[k:, k:] - 2 * np.outer(v, np.dot(v, A[k:, k:]))
        B[k:] = B[k:] - 2 * np.dot(v, np.dot(v, B[k:]))

    X = np.zeros(n)
    X[-1] = B[-1] / A[-1, -1]

    for k in range(n-2, -1, -1):
        X[k] = (B[k] - np.dot(A[k, k+1:], X[k+1:])) / A[k, k]

    return X

A = np.array([[1.0, 1.0, 1.0], [1.0, -2.0, 3.0], [0.0, 2.0, -1.0]])
B = np.array([1.0, -2.0, 1.0])

X = householder_solve(A, B)

print("Solução do sistema de equações:")
print(X)

