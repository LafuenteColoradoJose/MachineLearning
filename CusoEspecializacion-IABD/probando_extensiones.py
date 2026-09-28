
# * python-indent de Kevin Rose 

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
                print(z)