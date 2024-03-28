import matplotlib.pyplot as plt
import numpy as np


#  -----------------------------------------------------------------
# 1. Crear un gráfico de líneas simple
# a = [3, 4, 5, 6]
# b = [5, 6, 3, 4]

# plt.plot(a, b, color='red', linewidth=3, label='Línea')
# plt.legend()
# plt.show()

# -----------------------------------------------------------------
# 2. Crear un gráfico
# x = np.linspace(0, 2 * np.pi, 200)
# y = np.sin(x)

# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()

# -----------------------------------------------------------------

### DIAGRAMA DE LÍNEAS ###

#Definir los datos
x1 = [3, 4, 5, 6]
y1 = [5, 6, 3, 4]
x2 = [2, 5, 8]
y2 = [3, 4, 3]

#Configurar las características del gráfico
plt.plot(x1, y1, label='Línea 1', linewidth=2, color='blue')
plt.plot(x2, y2, label='Línea 2', linewidth=2, color='green')

#Definir título y nombres de ejes
plt.title('Diagrama de Líneas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar leyenda, cuadrícula y figura
plt.legend()
plt.grid()
plt.show()

### DIAGRAMA DE BARRAS ###

#Definir los datos
x1 = [0.25, 1.25, 2.25, 3.25, 4.25]
y1 = [10, 55, 80, 32, 40]
x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]

#Configurar las características del gráfico
plt.bar(x1, y1, label='Datos 1', width=0.5, color='lightblue')
plt.bar(x2, y2, label='Datos 2', width=0.5, color='orange')

#Definir título y nombres de ejes
plt.title('Gráfico de Barras')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar leyenda y figura
plt.legend()
plt.show()

### HISTOGRAMAS ###

#Definir los datos
a = np.array([22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 2, 102, 95, 85, 55, 110, 120, 70, 65, 55, 111, 115, 80, 75, 65, 54, 44, 43, 42, 48])
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#Configurar las características del gráfico
plt.hist(a, bins, histtype='bar', rwidth=0.8, color='lightgreen')

#Definir título y nombres de ejes
plt.title('Histograma')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar figura
plt.show()

### DIAGRAMA DE DISPERSIÓN ###

#Definir los datos
x1 = [0.25, 1.25, 2.25, 3.25, 4.25]
y1 = [10, 55, 80, 32, 40]
x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]

#Configurar las características del gráfico
plt.scatter(x1, y1, label='Datos 1', color='red')
plt.scatter(x2, y2, label='Datos 2', color='purple')

#Definir título y nombres de ejes
plt.title('Diagrama de Dispersión')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar leyenda y figura
plt.legend()
plt.show()

### DIAGRAMA DE PASTEL ###
#Definir los datos
sizes = [25, 20, 45, 10]
nombres = ['Manzanas', 'Plátanos', 'Mangos', 'Peras']
colores = ['orange', 'lightblue', 'lightgreen', 'pink']
explode = (0.1, 0, 0, 0)

#Configurar las características del gráfico
plt.pie(sizes, labels=nombres, colors=colores, explode=explode, shadow=True, autopct='%1.1f%%')

#Definir título
plt.title('Diagrama de Pastel')

#Mostrar figura
plt.show()





