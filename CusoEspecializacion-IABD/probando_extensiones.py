
# * Python Indent de Kevin Rose 

lista_3d = [
    [ # Página 1
        [1, 2, 3],  # Fila 1
        [4, 5, 6]   # Fila 2
    ],
    [ # Página 2
        [7, 8, 9],  # Fila 1
        [10, 11, 12]# Fila 2
    ]
]

for i in lista_3d:
    for x in i:
        for z in x:
            if(z == 6):
                print(z, "\n")
                
import numpy as np

array_3d = np.array(lista_3d)

print(array_3d.shape)
print(array_3d.size)
print(array_3d.max())
print(array_3d.min())
print(array_3d.mean())
print(array_3d.sum())

array_3d_ordenado_reves = -np.sort(-array_3d)
print('\n')
print("Ordenado al revés")
print(array_3d_ordenado_reves)