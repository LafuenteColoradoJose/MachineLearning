# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

a = np.array([1,2,3])
print('1D array')
print(a)

b = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int64)
print('2D array')
print(b)

# MATRICES VACIAS
unos = np.ones((3,4))
print('Matriz de unos')
print(unos)

ceros = np.zeros((3,4))
print('Matriz de ceros')
print(ceros)

aleatorios = np.random.random((2,2))
print('Matriz de aleatorios')
print(aleatorios)

vacia = np.empty((2,3))
print('Matriz vacía')
print(vacia)

full = np.full((2,2), 8)
print('Matriz llena')
print(full)

espacio1 = np.arange(0,30,5)
print('Arreglo con espacios')
print(espacio1)

espacio2 = np.linspace(0,2,5)
print('Arreglo con espacios')
print(espacio2)


identidad1 = np.eye(4,4)
print('Matriz identidad')
print(identidad1)

identidad2 = np.identity(4)
print('Matriz identidad')
print(identidad2)

# INSPECIOANR MATRICES

c = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print('Dimensiones de la matriz')
print(c.ndim)

d = np.array([(1,2,3)])
z = np.array([1,2,3])
print('Tipo de los datos')
print(d.dtype)

e = np.array([(1,2,3,4,5,6)])
print('Tamaño de la matriz')
print(d.size)
print(z.size)
print('Forma de la matriz')
print(d.shape)
print(z.shape)


# CAMBIAR FORMA DE MATRICES

f = np.array([(8,9,10), (11,12,13)])
print('Matriz original')
print(f)
print('Matriz transpuesta')
f = f.reshape(3,2)
print(f)

# extraer un solo valor
print('Extraer un solo valor')
print(f[0,1])

# extraer todos los valores de todas las filas ubicados en la columna 1

print('Extraer todos los valores de todas las filas ubicados en la columna 1')
print(f[0:,1])

# OPERACUIONES MATEMATICAS
g = np.array([2,4,8])
print('Suma de los elementos')
print(g.sum())
print('Valor máximo')
print(g.max())
print('Valor mínimo')
print(g.min())
print('Promedio')
print(g.mean())

# calcular la raiz cuadrada y la desviación estandar
h = np.array([(1,2,3), (4,5,6)])
print('Raiz cuadrada')
print(np.sqrt(h))
print('Desviación estandar')
print(np.std(h))


# OPERACIONES CON MATRICES
i = np.array([(1,2,3), (4,5,6)])
j = np.array([(1,2,3), (4,5,6)])

print('Suma de matrices')
print(i+j)
print('Resta de matrices')
print(i-j)
print('Multiplicación de matrices')
print(i*j)
print('División de matrices')
print(i/j)










