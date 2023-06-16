import numpy as np

A = np.array([[1.0, -1.0, -1], [1.0, 0.0, 1.0], [1.0, -1.0, 3.0]])
B = np.array([-5.0/3.0, 3.0/4.0, 7.0/3.0])

# Passo 1: Normalizar a primeira coluna de A para obter o primeiro vetor da matriz Q
q1 = A[:, 0] / np.linalg.norm(A[:, 0])

# Passo 2: Calcular o segundo vetor da matriz Q
v2 = A[:, 1] - np.dot(q1, A[:, 1]) * q1
q2 = v2 / np.linalg.norm(v2)

# Passo 3: Calcular o terceiro vetor da matriz Q
v3 = A[:, 2] - np.dot(q1, A[:, 2]) * q1 - np.dot(q2, A[:, 2]) * q2
q3 = v3 / np.linalg.norm(v3)

# Passo 4: Construir a matriz Q
Q = np.column_stack((q1, q2, q3))

# Passo 5: Calcular a matriz R
R = np.dot(Q.T, A)

# Passo 6: Resolver o sistema linear equivalente RX = Q^T * B para encontrar o vetor X
X = np.linalg.solve(R, np.dot(Q.T, B))

print("Matriz Q:")
print(Q)
print("\nMatriz R:")
print(R)
print("\nVetor X:")
print(X)

