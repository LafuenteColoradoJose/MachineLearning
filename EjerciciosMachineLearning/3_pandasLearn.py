import numpy as np
import pandas as pd

# DataFrame básico
data = np.array([['','Col1','Col2'], ['Fila1',11,22], ['Fila2',33,44]])
print('DATAFRAME')
print(pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]))
print('\n')


df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print('DataFrame básico')
print(df)
print('\n')

series = pd.Series({'Agerntina': 'Buenos Aires', 'Chile': 'Santiago', 'Colombia': 'Bogotá', 'Perú': 'Lima'})
print('SERIES:')
print(series)
print('\n')

# Forma del DataFrame
print('Forma del DataFrame')
print(df.shape)
print('\n')

# Altura del DataFrame
print('Altura del DataFrame')
print(len(df.index))
print('\n')

# Columnas del DataFrame
print('Columnas del DataFrame')
print(df.columns)
print('\n')

#Estadaísticas del DataFrame
print('Estadísticas del DataFrame')
print(df.describe())
print('\n')

#Media de las columnas del DataFrame
print('Media de las columnas del DataFrame')
print(df.mean())
print('\n')

#Media de las filas del DataFrame
print('Media de las filas del DataFrame')
print(df.mean(1))
print('\n')

#Correlación del DataFrame
print('Correlación del DataFrame')
print(df.corr())
print('\n')

#Conteo de los valores del DataFrame
print('Conteo de los valores del DataFrame')
print(df.count())
print('\n')

# Valor máximo de las columnas del DataFrame
print('Valor máximo de las columnas del DataFrame')
print(df.max())
print('\n')

# Valor mínimo de las columnas del DataFrame
print('Valor mínimo de las columnas del DataFrame')
print(df.min())
print('\n')

# Mediana de las columnas del DataFrame
print('Mediana de las columnas del DataFrame')
print(df.median())
print('\n')

# Desviación estándar de las columnas del DataFrame
print('Desviación estándar de las columnas del DataFrame')
print(df.std())
print('\n')

# Seleccionar ka primea columna del DataFrame
print('Seleccionar la primera columna del DataFrame')
print(df[0])
print('\n')

# Seleccionar dos columnas del DataFrame
print('Seleccionar dos columnas del DataFrame')
print(df[[0, 1]])
print('\n')

# Selcciopnar el valor de la primera fila y la última columna del DataFrame
print('Seleccionar el valor de la primera fila y la última columna del DataFrame')
print(df.iloc[0][2])
print('Selecionar el valor de la última fila y la primera columna del DataFrame')
print(df.iloc[2][0])
print('\n')

# Seleccionar los valores de la primera fila del DataFrame
print('Seleccionar los valores de la primera fila del DataFrame')
print(df.loc[0])
print('\n')
print(df.iloc[0,:])

# Importar Exportar datos
# pd.read_tipodearchivo('nombrearchivo')
# pd.to_tipodearchivo('nombrearchivo', index=False)


# data = pd.read_csv('data.csv')

# Verificar si hay datos nulos en el DataFrame
print('Verificar si hay datos nulos en el DataFrame')
print(df.isnull())
print('\n')

# Suma de los valores nulos en el DataFrame
print('\033[31m' + 'Suma de los valores nulos en el DataFrame' + '\033[0m')
print('------------------------------------------------------------------') 
print(df.isnull().sum())
print('\n')

# Eliminar los valores nulos en el DataFrame
# pd.dropna()
# df.dropna(axis=1)

# Rellenar los valores nulos en el DataFrame
# pd.fillna(valor)













