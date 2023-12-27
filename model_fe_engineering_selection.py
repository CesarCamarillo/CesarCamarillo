# # Tema 2: Ingeniería de características y selección de variables en SKlearn

# ## Objetivo del Ciclo 2 ID:
# El objetivo del Ciclo 2 ID es conocer sobre la ingenieria de características y selección de variables como paramétros en la construcción de los modelos de KNN

# ### 2.1 Importación de librerías
# * Nuestro primer paso consiste en importar todas las librerías que utilizaremos en este código, las cuales se enlistan a continuación:
#     * numpy y pandas (para el manejo de la base de datos de las solicitudes de las bicicletas)
#     * matplotlib (para generar gráficos que nos permitan visualizar la información compartida)
#     * Sklearn (para la generación de los diferentes modelos de KNN Vecinos, utilizando sus siguientes dos módulos:)
#         * KNeighborsRegressor (crea modelos de regresión KNN, lo que nos permitirá generar predicciones de la demanda de bicicletas)
#         * mean_squared_error (métrica de medición de los errores de la predicción, en contraste con los datos reales)
#         * Pipeline
#         * LabelEncoder,OrdinalEncoder,LabelBinarizer y OneHotEncoder (para transformación numérica de variables categóricas)
#         * StandardScaler y MinMaxScaler (para normalización/estandarización de variables)
#         * PowerTransformer (para cambios a distribución normal de variables)
#         * ColumnTransformer (para combinación de procesos de transformación a variables)
#         * VarianceThreshold  (para identifiación de variables relevantes para el modelo)
#     * pickle (para exportar un modelo)


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder,OrdinalEncoder,LabelBinarizer,OneHotEncoder)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
import pickle


# ### 2.2 Lectura de la base de datos
# Leemos la base de datos proporcionada, utilizando la librería de pandas.
# Dentro de la función de read_csv hemos definido el parámetro de "encoding" como "latin1" para que pueda leer caracteres especiales, ya sea en los nombres de las variables y/o en los valores de estas


#Leemos la base de datos proporcionada
bd_bikepro=pd.read_csv(r"C:\Users\cjcamper.AFIRME\Documents\Certificacion AI Core\Primer_reto\Recursos\Data\SeoulBikeData.csv",encoding='latin1')


# Hacemos un head para visualizar las primeras filas de nuestra base
bd_bikepro.head()



# ### 2.3 Modificación de nombres y valores de variables

# #### Ajustes a realizar a los nombres de nuestras columnas:
# Realizamos un "one-liner" (código en una sola línea) para este bucle de iteración sobre las columnas y darles un mejor formato, partiendo por los siguientes pasos:
 
# 1. Primero vamos a iterar sobre cada columna del dataset, la cual llamaremos **"col"**
# 2. Luego vamos a convertir cada columna en minúsculas con la función **lower**
# 3. Al ver los nombres de variables como **"Visibility"**, cuyos nombres están separados del primer paréntesis (en donde tenemos sus respectivas escalas), sustitumos ese paréntesis por un paréntesis sin espacio con la función **replace**
# 4. Después realizamos un **split** entre los paréntesis, que nos separará los nombres de las columnas de los paréntesis (ya sin espacio) , para solo quedarnos con el nombre de la variable sin su escala
# 5. En el siguiente paso separamos cada palabra del nombre de la variable, nuevamente usando la función **split**, ahora separando por espacios
# 6. Finalmente, con el método **join**, juntaremos con un guion bajo como delimitador cada una de las palabras
 


#las nuevas variables se guardarán en el elemento "new_columns"
new_columns=["_".join(col.lower().replace(' (','(').split('(')[0].split(' ')) for col in bd_bikepro.columns]


#ahora asignamos el elemento "new_columns" a nuestra base de datos
bd_bikepro.columns=new_columns


# ### Modificación de formato para los valores de columnas
# La variable "date" juega un rol fundamental ya que nos permite ordenar nuestros datos, por lo que necesitamos que esté en el formato correcto de fecha



#convertimos en formato fecha la variable de "date"
bd_bikepro['date'] = pd.to_datetime(bd_bikepro['date'], format='%d/%m/%Y')



#Seleccionaremos nuestras columnas del clima descartando todas aquellas variables que no estén relacionadas al clima
weather_cols=[col for col in bd_bikepro.columns if col not in ['date','rented_bike_count','hour','seasons','holiday','functioning_day']]
target_col = ['rented_bike_count']



# ### 2.4 Generación de nuevas variables
# Generaremos una variable que nos indique el mes de la fecha, así como otra variable que señale si el día de demanda de bicicletas fue en fin de semana



#empezamos generando la variable del mes del año
bd_bikepro['month_day']=bd_bikepro['date'].dt.month.astype('category')



#generamos la bandera que nos indique si el día es fin de semana
bd_bikepro['flag_day_weekend'] = np.where(bd_bikepro['date'].dt.weekday> 4,1,0)



# ### 2.5 Ordenamiento de datos
# Tal como se comentó, ordenaremos nuestra información por fecha y hora. Este paso es muy importante ya que necesitamos considerar toda la tendencia histórica que ha habido en la demanda de bicicletas, ya que este dato será vital para la construcción del modelo y la predicción que nos dará

# La función "sort_values", definiendo los campos para ordenación, nos ayudará a ordenar nuestros datos, tomando el paramétro "inplace" como **True** para que realiza la ordenación en nuestro dataset


bd_bikepro.sort_values(by=['date','hour'],inplace=True)



# ### 2.6 Transformación de variables categóricas
# Para este paso  vamos a convertir nuestras variables categóricas a numéricas, usando diferentes transformaciones dependiendo de los valores de cada variabe. Las variables categóricas a transformar son:
# * seasons
# * functioning_day
# * holiday
# * flag_day_weekend
# * month_day


#Las variables "seasons" ,"functioning_day" "holiday", "flag_day_weekend" y "month_day" serán transformadas por OneHotEncoder
categorical_pipeline = Pipeline([('categorical', OneHotEncoder(handle_unknown='ignore'))])


#La variable "month_day" será transformada por OrdinalEncoder
ordinal_pipeline = Pipeline([('ordinal', OrdinalEncoder())])



# ### 2.7 Cambios en la distribución de las variables

# Al hacer un histograma de cada una de las variables numéricas del clima observamos que ninguna sigue una distribución normal, por lo que se tienen que transformar los valores de estas


bd_bikepro[weather_cols].hist()



#como tercer paso decidimos transformar las variables numéricas del clima por un proceso de estandarización
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, r_regression

numerical_pipeline = Pipeline([
    ('standar_scaler', StandardScaler())
])


# Dos posibles métodos para transformar estas variables son la transformación de Box-Cox y Yeo-Johnson.
# Partiendo que la transformación de Box-Cox no permite valores igual o menores a 0, hacemos una agrupación mínima por variables del clima para comprobar lo anterior



bd_bikepro[weather_cols].min()
#se observan que todas las variables cumplen lo anterior, por lo que tenemos que utilizar la transformación de Yeo-Johnson


# Generamos un pipeline para la estandarización de las variables que presenten una distribución diferente a la normal
transformation_pipeline = Pipeline([('transformation', PowerTransformer(method='yeo-johnson'))])



# ### 2.8 Normalización/estandarización de variables
 
# Al tener variables con diferentes unidades, es recomendable que los valores tengan una misma magnitud. Dos métodos para lograr esto son la estandarización



#generamos un pipeline para la estandarización de las variables
numerical_pipeline = Pipeline([
    ('standar_scaler', StandardScaler())
])


# ### 2.9 Selección de variables
# El objetivo de este pipeline es descartar variables cuya varianza sea muy poca o nula, esto al no tener una influencia en el modelo
selector_pipeline = Pipeline([('selector', VarianceThreshold())])


# ### 2.10 Selección de variables predictora y variable objetivo
# Antes de generar nuestro conjunto de entrenamiento y prueba, es importante definir cuáles variables tomaremos en cuenta para la construcción de nuestro modelo, así como la variable que intentaremos predecir (cantidad de bicicletas rentadas)


#Entendiendo que todas nuestras variables participarán en el modelo (con sus respectivas transformaciones), generamos una nueva lista con todas las variables, excluyendo la fecha de la demanda de bicicletas, así como la variable objetivo (rented_bike_count)
all_cols=list(bd_bikepro.columns.difference(['date','rented_bike_count']))


#Hacemos una lista de variables relacionadas con el tiempo con los días
days_cols=["functioning_day", "holiday" ,"flag_day_weekend","month_day","seasons"]



# ### 2.11 Creación de conjunto de entrenamiento y prueba

# Para la ejecución de los modelos de aprendizaje automático es importante generar dos conjuntos, uno de entrenamiento y otro de prueba, ya sea para las variables que predecirán los valores, así como la variable objetivo a predecir, respectivamente

# * En el conjunto de **entrenamiento** construiremos nuestro modelo de KNN
# * En el conjunto de **prueba** buscamos comparar los resultados que se predijeron contra los resultados reales

# La demanda de bicicletas tiene un componente temporal, es decir, la demanda ha sido influida por la hora y día en la que se rentaron las bicicletas, por lo que es importante que nuestros conjuntos de entrenamiento y prueba sigan captando las tendencias mostradas. Debido a lo anterior, generaremos nuestros conjuntos dividiendo el dataset original desde determinado registro


# Datos de entrenamiento
x_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][all_cols]
y_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][target_col]

# Datos de test
x_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][all_cols]
y_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][target_col]


# ### 2.12 Ejecución del modelo
# Usando la función ColumnTransformer realizaremos todas nuestras transformaciones generadas en pasos previas, todas en una misma instancia

pre_processor = ColumnTransformer([
    ('numerical', numerical_pipeline, weather_cols)
    ,('categorical', categorical_pipeline, days_cols)
    ,('ordinal',ordinal_pipeline,['hour'])
    ,('transformation',transformation_pipeline,weather_cols)
], remainder='passthrough')


# ### Pasos de ejecución del modelo
 
# 1. Transformaciones de variables
# 2. Filtrado de variables que no sean relevantes en el modelo, según su varianza
# 3. Ejecución del modelo de KNN con 30 vecinos



#nuestro modelo tendrá la instancia de transformaciones de las variables, luego el descarte de varables según su varianza
pipeline_final = Pipeline([
    ('transform', pre_processor),
    ('selector',selector_pipeline),
    ('model', KNeighborsRegressor(n_neighbors=30))
])


# Ejecutamos el modelo, midiendo su error, así como su precisión

pipeline_final.fit(x_train, y_train)
y_test_pred = pipeline_final.predict(x_test)
error=mean_squared_error(y_test,y_test_pred)
#generamos el score entre los datos de test
score=pipeline_final.score(x_test,y_test)



# ### 2.13 Exportación del modelo
# Usando la librería pickle vamos a exportar nuestro modelo, para que en ocasiones futuras podamos utilizarlo

#publicamos el modelo
import pickle
pickle.dump(pipeline_final, open('model_fe_engineering_selection.pkl', 'wb'))






