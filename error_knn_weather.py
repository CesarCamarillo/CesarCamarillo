#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


#Leemos la base de datos proporcionada
bd_bikepro=pd.read_csv(r"C:\Users\cjcamper.AFIRME\Documents\Certificacion AI Core\Primer_reto\Recursos\Data\SeoulBikeData.csv",encoding='latin1')


# In[4]:


#Hacemos un head para visualizar las primeras filas de nuestra base
bd_bikepro.head()


# In[5]:


[col.lower().replace(' (','(').split('(')[0] for col in bd_bikepro.columns]


# In[6]:


#Realizamos un "one-liner para este bucle de iteración sobre las columnas y darles un mejor formato"
#Primero vamos a iterar sobre cada columna del dataset
#Luego vamos a convertir cada columna (definida con el nombre de "col") en minúsculas
#Al ver los nombres de variables como "Visibility", cuyos nombres están separados del primer paréntesis, sustitumos ese paréntesis por uno sin espacio
#Después realizamos un split entre los paréntesis, para solo quedarnos con el nombre de la variable sin su escala
#En siguiente paso separamos cada palabra del nombre de la variable, para que finalmente, con el método join, puedan juntarse con un guion bajo como delimitador
new_columns=["_".join(col.lower().replace(' (','(').split('(')[0].split(' ')) for col in bd_bikepro.columns]


# In[7]:


#Podemos ver los nuevos nombres de las variables
bd_bikepro.columns=new_columns
bd_bikepro.head(5)


# In[8]:


#convertimos en formato fecha la variable de "date"
bd_bikepro['date'] = pd.to_datetime(bd_bikepro['date'], format='%d/%m/%Y')


# In[9]:


#Seleccionaremos nuestras columnas del clima descartando todas aquellas variables que no estén relacionadas al clima
weather_cols=[col for col in bd_bikepro.columns if col not in ['date','rented_bike_count','hour','seasons','holiday','functioning_day']]
target_col = ['rented_bike_count']


# In[10]:


#Antes de generar nuestra base con la que se conformarán la base de entrenamiento y test vamos a ordenar por fecha y hora
bd_bikepro.sort_values(by=['date','hour'],inplace=True)


# In[11]:


# Datos de entrenamiento
x_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][weather_cols]
y_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][target_col]

# Datos de test
x_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][weather_cols]
y_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][target_col]


# In[12]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# In[13]:


#definimos nuestro vector de K, en el que iteraremos para encontrar el mejor parámetro para nuestro modelo
vect_k=[3,5,10,15,20,50,100,300,500,1000]


# In[14]:


#definimos un vector donde estaremos depositando nuestros errores
vect_error=[]
#definimos un vector donde depositaremos el accuracy entre cada iteración de k
vect_score=[]
for i in range(len(vect_k)):
    model = KNeighborsRegressor(n_neighbors=vect_k[i])
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    error=mean_squared_error(y_test,y_test_pred)
    vect_error.append(error)
    #generamos el score entre los datos de test
    vect_score.append(model.score(x_test,y_test))


# In[15]:


#generamos un gráfico para comparar los errores para cada elemento de K
plt.figure(figsize=(25,5))

plt.plot(vect_k,vect_error)
plt.title("Gráfico Comparativo de error entre elementos de K")
plt.xlabel("K")
plt.ylabel("MSE")
plt.axvline(x=20,color='red',linestyle='--')


# In[16]:


#hacemos un nuevo gráfico haciendo zoom hasta el elemento 50 de los k vecinos
plt.figure(figsize=(15,5))

plt.plot(vect_k[:6],vect_error[:6])
plt.title("Gráfico Comparativo de error entre elementos de K (hasta 50)")
plt.xlabel("K")
plt.ylabel("MSE")
plt.axvline(x=20,color='red',linestyle='--')
for x,y in zip(vect_k[:6],vect_error[:6]):
    plt.text(x,y,s=(x,'{:,.2f}'.format(y)))
#de esta manera se observa mejor cómo en el valor 20 del vector de vecinos se obtiene el menor valor de error, así como después de ese valor el error tiende a incrementarse


# In[20]:


#
plt.figure(figsize=(15,5))

model = KNeighborsRegressor(n_neighbors=20)
model.fit(x_train, y_train)

#generamos las prediccinones con los K vecinos en 20, tanto para el conjunto de test como el conjunto de train
y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)

error_test=mean_squared_error(y_test,y_test_pred)
error_train=mean_squared_error(y_train,y_train_pred)

plt.plot(y_test.reset_index(drop=True))
plt.plot(list(y_test_pred))
plt.title('Figura 3. Datos de entrenamiento - Valores reales vs valores predccion')
plt.legend(["Actual", "Predicted"])
plt.show()


# In[22]:


#el error en el conjunto de entrenamiento fue 201694.03053339708
#el error en el conjunto de test fue 318367.1524027103
#se mantiene la tendencia observada a que el error en el conjunto de test no es parecido al error en el conjunto de entrenamiento

