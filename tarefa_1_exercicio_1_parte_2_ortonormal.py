import numpy as np

def normalize_columns(A):
    norms = np.linalg.norm(A, axis=0)
    A_normalized = A / norms
    return A_normalized

# Exemplo de uso
Q = np.array([[1, -1.166667, 0.27586207], [2, -0.3333333, -0.20689655], [1, 1.8333333, 0.13793103]])

A_normalized = normalize_columns(Q)

print("Matriz ortogonal Q:")
print(Q)

print("\nMatriz ortonormal Â:")
print(A_normalized)

# Verificando se Â é realmente ortonormal
print("\nProduto interno das colunas de Â:")
print(np.dot(A_normalized[:, 0], A_normalized[:, 1]))
print(np.dot(A_normalized[:, 0], A_normalized[:, 2]))
print(np.dot(A_normalized[:, 1], A_normalized[:, 2]))

# Verificando se Â é ortonormal
print("\nNorma das colunas de Â:")
print(np.linalg.norm(A_normalized[:, 0]))
print(np.linalg.norm(A_normalized[:, 1]))
print(np.linalg.norm(A_normalized[:, 2]))

