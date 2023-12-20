#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn


# In[2]:


#Leemos la base de datos proporcionada
bd_bikepro=pd.read_csv(r"C:\Users\cjcamper.AFIRME\Documents\Certificacion AI Core\Primer_reto\Recursos\Data\SeoulBikeData.csv",encoding='latin1')


# In[3]:


#Hacemos un head para visualizar las primeras filas de nuestra base
bd_bikepro.head()


# In[4]:


[col.lower().replace(' (','(').split('(')[0] for col in bd_bikepro.columns]


# In[5]:


#Realizamos un "one-liner para este bucle de iteración sobre las columnas y darles un mejor formato"
#Primero vamos a iterar sobre cada columna del dataset
#Luego vamos a convertir cada columna (definida con el nombre de "col") en minúsculas
#Al ver los nombres de variables como "Visibility", cuyos nombres están separados del primer paréntesis, sustitumos ese paréntesis por uno sin espacio
#Después realizamos un split entre los paréntesis, para solo quedarnos con el nombre de la variable sin su escala
#En siguiente paso separamos cada palabra del nombre de la variable, para que finalmente, con el método join, puedan juntarse con un guion bajo como delimitador
new_columns=["_".join(col.lower().replace(' (','(').split('(')[0].split(' ')) for col in bd_bikepro.columns]


# In[6]:


#Podemos ver los nuevos nombres de las variables
bd_bikepro.columns=new_columns
bd_bikepro.head(5)


# In[7]:


#convertimos en formato fecha la variable de "date"
bd_bikepro['date'] = pd.to_datetime(bd_bikepro['date'], format='%d/%m/%Y')


# In[8]:


#Seleccionaremos nuestras columnas del clima descartando todas aquellas variables que no estén relacionadas al clima
weather_cols=[col for col in bd_bikepro.columns if col not in ['date','rented_bike_count','hour','seasons','holiday','functioning_day']]
target_col = ['rented_bike_count']


# In[9]:


#Antes de generar nuestra base con la que se conformarán la base de entrenamiento y test vamos a ordenar por fecha y hora
bd_bikepro.sort_values(by=['date','hour'],inplace=True)


# In[10]:


# Datos de entrenamiento
x_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][weather_cols]
y_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][target_col]

# Datos de test
x_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][weather_cols]
y_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][target_col]


# In[11]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# In[12]:


from sklearn.preprocessing import PolynomialFeatures

model_pol = PolynomialFeatures(degree=2)


# In[13]:


model_pol.fit(x_train[weather_cols])


# In[14]:


#tomamos el conjunto de entrenamiento con únicamente las variables númericas del clima para aplicarles una transformación polinómica de grado 2
x_train_pol = pd.DataFrame(model_pol.transform(x_train[weather_cols]),columns=model_pol.get_feature_names(weather_cols))


# In[15]:


#hacemos la misma transformación para el conjunto de prueba
x_test_pol = pd.DataFrame(model_pol.transform(x_test[weather_cols]),columns=model_pol.get_feature_names(weather_cols))


# In[16]:


#para motivos de simplicidad eliminamos para ambos dataset la primera columna ya que solo tenemos valores constantes y no nos otorga valor
x_test_pol=x_test_pol[x_test_pol.columns[1:]]
x_train_pol=x_train_pol[x_train_pol.columns[1:]]


# In[17]:


#definimos nuestro vector de K, en el que iteraremos para encontrar el mejor parámetro para nuestro modelo
vect_k=[5,10,15,20,30,50,100,200,400,500]


# In[18]:


from sklearn.pipeline import Pipeline


# In[19]:


#definimos un vector donde estaremos depositando nuestros errores
vect_error=[]
#definimos un vector donde depositaremos el accuracy entre cada iteración de k
vect_score=[]

#realizaremos el mismo ejercicio del ciclo anterior, solo ahora utilizando los datos de entrenamiento y pruebas, transformados por el polinomio 2
for i in range(len(vect_k)):
    #nuestros estimadores para el modelo de KNN contarán con la transformación polinómica de los conjuntos de entrenamiento y prueba
    #igualmente iterará entre cada elemento de K para generar diferentes modelos
    estimators = [('polinomical_features', PolynomialFeatures(degree=2)),('knn_model', KNeighborsRegressor(n_neighbors=vect_k[i]))]
    
    model = Pipeline(steps=estimators)
    #al ya estar contemplado la transformación polinómica dentro de los estimadores del modelo de KNN se utilizarán las bases originales de entrenamiento y prueba
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    error=mean_squared_error(y_test,y_test_pred)
    vect_error.append(error)
    #generamos el score entre los datos de test
    vect_score.append(model.score(x_test,y_test))


# In[21]:


plt.figure(figsize=(25,5))

plt.plot(vect_k,vect_error)
plt.title("Gráfico Comparativo de error entre elementos de K")
plt.xlabel("K")
plt.ylabel("MSE")
plt.axvline(x=50,color='red',linestyle='--')


# In[22]:


#simulando el ejercicio con todos los elementos de K se obtiene que el mejor modelo utiliza 50 vecinos, teniendo un error en el conjunto de test de 327439.009
#comparado con el modelo del ciclo pasado, cuyo valor de K Vecinos fue 20, se muestra un ligero aumento en su correspondinente error en el conjunto de prueba (318367.15)


# In[23]:


#de igual manera que el modelo del ciclo pasado, el error en el conjunto de prueba es significativamente más grande que el correspondiente en el conjunto de entrenamiento, lo que indica un sobreajuste en el modelo
#si se usara el modelo así nos enfrentaríamos al riesgo de que el modelo no genere predicciones precisas para datos del futuro, ya que se ajustó muy bien a las tendencias en el dataset, por lo que no sabría detectar nuevos patrones


# In[24]:


#al entender que aún no podemos utilizar el modelo, vamos a generar unas simulaciones basados en lass siguientes instancias de feature engineering
# 1- vamos a generar nuevas variables a partir de las variables que ya tenemos
# 2.- transformaremos nuestras variables categóricas en variables numéricas
# 3.- normalizaremos/estandarizaremos las variables relacionadas al clima
# 4.- modificaremos la distribución de los valores de las variables numéricas


# In[49]:


#empezamos generando la variable del mes del año
bd_bikepro['month_day']=bd_bikepro['date'].dt.month.astype('category')


# In[26]:


#generamos la bandera que nos indique si el día es fin de semana
bd_bikepro['flag_day_weekend'] = np.where(bd_bikepro['date'].dt.weekday> 4,1,0)


# In[27]:


#para el paso dos vamos a convertir nuestras variables categóricas a numéricas
#usaremos diferentes transformaciones dependiendo de los valores de cada variabe
from sklearn.preprocessing import (LabelEncoder,OrdinalEncoder,LabelBinarizer,OneHotEncoder)


# In[30]:


#Las variables "seasons" ,"functioning_day" "holiday" y "flag_day_weekend" serán transformadas por OneHotEncoder
categorical_pipe = Pipeline([('categorical', OneHotEncoder(handle_unknown='ignore'))])


# In[31]:


#La variable "month_day" será transformada por OrdinalEncoder
ordinal_pipe = Pipeline([('ordinal', OrdinalEncoder())])


# In[33]:


#al hacer un histograma de cada una de las variables numéricas del clima observamos que ninguna sigue una distribución normal, por lo que se tienen que transformar los valores
bd_bikepro[weather_cols].hist()


# In[126]:


#como tercer paso decidimos transformar las variables numéricas del clima por un proceso de estandarización
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, r_regression

numerical_pipe = Pipeline([
    ('standar_scaler', StandardScaler())
])


# In[95]:


#entendiendo que la transformación de Box-Cox no permite valores igual o menores a 0, hacemos una agrupación mínima por variables del clima para comprobar lo anterior
bd_bikepro[weather_cols].min()
#se observan que todas las variables cumplen lo anterior, por lo que tenemos que utilizar la transformación de Yeo-Johnson


# In[96]:


#generamos una instancia para la transformación por Yeo-Johnson
from sklearn.preprocessing import PowerTransformer
transformation_pipe = Pipeline([('transformation', PowerTransformer(method='yeo-johnson'))])


# In[97]:


from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold


# In[98]:


#generamos una instancia para la selección de variables cuya variance no cumplan el threshold
#esto sirve mucho para descartar variables que, al tener poca o nula varianza, podemos considerarlas como constantes, y no aportarían mucho al modelo
selector_pipe = Pipeline([('selector', VarianceThreshold())])


# In[121]:


#entendiendo que todas nuestras variables participarán en el modelo (con sus respectivas transformaciones), generamos una nueva lista con todas las variables
#la lista la generamos partiendo de las columnas de nuestro dataset original, eliminando las variables de "date" y "rented_bike_count"
all_cols=list(bd_bikepro.columns.difference(['date','rented_bike_count']))


# In[122]:


# Datos de entrenamiento
#hacemos una lista de variables relacionadas con el tiempo
time_cols=["functioning_day", "holiday" ,"flag_day_weekend"]
x_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][all_cols]
y_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][target_col]

# Datos de test
x_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][all_cols]
y_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][target_col]


# In[123]:


pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols)
    ,('categorical', categorical_pipe, ['seasons']+time_cols)
    ,('ordinal',ordinal_pipe,['hour'])
    ,('transformation',transformation_pipe,weather_cols)
], remainder='passthrough')


# In[124]:


#nuestro modelo tendrá la instancia de transformaciones de las variables, luego el descarte de varables según su varianza
pipe_final = Pipeline([
    ('transform', pre_processor),
    ('selector',selector_pipe),
    ('model', KNeighborsRegressor(n_neighbors=5))
])


# In[130]:


#definimos un vector donde estaremos depositando nuestros errores
vect_error=[]
#definimos un vector donde depositaremos el accuracy entre cada iteración de k
vect_score=[]

from sklearn.feature_selection import SelectKBest, r_regression

pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols)
    ,('categorical', categorical_pipe, ['seasons']+time_cols+['month_day'])
    ,('ordinal',ordinal_pipe,['hour'])
    ,('transformation',transformation_pipe,weather_cols)
], remainder='passthrough')

for i in range(len(vect_k)):
    
    #igualmente iterará entre cada elemento de K para generar diferentes modelos
    
    pipe_final = Pipeline([('transform', pre_processor),('selector',selector_pipe),('model', KNeighborsRegressor(n_neighbors=vect_k[i]))])
    
    
    
    pipe_final.fit(x_train, y_train)
    y_test_pred = pipe_final.predict(x_test)
    error=mean_squared_error(y_test,y_test_pred)
    vect_error.append(error)
    #generamos el score entre los datos de test
    vect_score.append(pipe_final.score(x_test,y_test))
    


# In[141]:


#revisando cada iteración vemos que el modelo con menor error fue el error con 5 K Vecinos (136026.34)
#haremos una última iteración, ahora agregando la instancia de SelectKBest


# In[142]:


numerical_pipe = Pipeline([
    ('standar_scaler', StandardScaler()),
    ('select_k_best',SelectKBest(r_regression, k=4) )
])


# In[143]:


#definimos un vector donde estaremos depositando nuestros errores
vect_error=[]
#definimos un vector donde depositaremos el accuracy entre cada iteración de k
vect_score=[]

from sklearn.feature_selection import SelectKBest, r_regression

pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols)
    ,('categorical', categorical_pipe, ['seasons']+time_cols+['month_day'])
    ,('ordinal',ordinal_pipe,['hour'])
    ,('transformation',transformation_pipe,weather_cols)
], remainder='passthrough')

for i in range(len(vect_k)):
    
    #igualmente iterará entre cada elemento de K para generar diferentes modelos
    
    pipe_final = Pipeline([('transform', pre_processor),('selector',selector_pipe),('model', KNeighborsRegressor(n_neighbors=vect_k[i]))])
    
    
    
    pipe_final.fit(x_train, y_train)
    y_test_pred = pipe_final.predict(x_test)
    error=mean_squared_error(y_test,y_test_pred)
    vect_error.append(error)
    #generamos el score entre los datos de test
    vect_score.append(pipe_final.score(x_test,y_test))
    


# In[145]:


#nuestro modelo con menor error sigue siendo el correspondiente a 5 Vecinos, sin embargo, hemos podido reducir el error a 129421.02


# In[156]:


#por último hacemos una comparativa entre los modelos con 5 vecinos versus el modelo de 30 vecinos
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15,5))

plt.plot(vect_k[:5],vect_error[:5])
plt.title("Gráfico Comparativo de error entre elementos de K (hasta 50)",color="black")
plt.xlabel("K",color="black")
plt.ylabel("MSE",color="black")
plt.axvline(x=5,color='red',linestyle='--')
plt.axvline(x=30,color='green',linestyle='--')
for x,y in zip(vect_k[:6],vect_error[:5]):
    plt.text(x,y,s=(x,'{:,.2f}'.format(y)),fontsize=10)


# In[157]:


#publicamos el modelo
import pickle
pickle.dump(pipe_final, open('model_fe_engineering_selection.pkl', 'wb'))


# In[ ]:




