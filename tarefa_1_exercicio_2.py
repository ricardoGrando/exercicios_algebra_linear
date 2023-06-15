import sympy as sp

# Definindo a variável simbólica
x = sp.Symbol('x')

# Definindo as funções f1(x) e f2(x)
f1 = 1
f2 = x

# Calculando as funções ortogonalizadas g1(x) e g2(x) usando o processo de Gram-Schmidt
g1 = f1
g2 = f2 - sp.integrate(f2 * g1, (x, 0, 1)) * g1

# Normalizando as funções g1(x) e g2(x)
g1 = g1 / sp.sqrt(sp.integrate(g1**2, (x, 0, 1)))
g2 = g2 / sp.sqrt(sp.integrate(g2**2, (x, 0, 1)))

print("As funções ortonormais g1(x) e g2(x) são:")
print(g1)
print(g2)
