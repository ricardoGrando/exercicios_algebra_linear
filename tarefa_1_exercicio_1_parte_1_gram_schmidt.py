import numpy as np

np.set_printoptions(suppress = True)

def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros_like(A)
    
    for i in range(n):
        Q[:, i] = A[:, i]
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, j], A[:, i]) / np.dot(Q[:, j], Q[:, j]) * Q[:, j]
            
    return Q

# Exemplo de uso
A = np.array([[1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
Q = gram_schmidt(A)

print("Matriz A:")
print(A)

print("\nMatriz ortogonal Q:")
print(Q)

# Verificando se Q é realmente ortogonal
print("\nProduto interno das colunas de Q:")
print(np.dot(Q[:, 0], Q[:, 1]))
print(np.dot(Q[:, 0], Q[:, 2]))
print(np.dot(Q[:, 1], Q[:, 2]))

# Verificando se Q é ortonormal
print("\nNorma das colunas de Q:")
print(np.linalg.norm(Q[:, 0]))
print(np.linalg.norm(Q[:, 1]))
print(np.linalg.norm(Q[:, 2]))


