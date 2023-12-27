
# # Tema 1: Fundamentos de Modelos de Aprendizaje de Maquina

# ## Objetivo del Ciclo 1 ID:
# El objetivo del Ciclo 1 ID es conocer las bases de los modelos de aprendizaje de maquina, empezando por la aplicacion del Modelo de KNN para predecir la demanda de bicicletas de BikePro

# ### 1.1 Importacion de librerias
# * Nuestro primer paso consiste en importar todas las libreri≠as que utilizaremos en este codigo, las cuales se enlistan a continuacion:
#     * numpy y pandas (para el manejo de la base de datos de las solicitudes de las bicicletas)
#     * matplotlib (para generar graficos que nos permitan visualizar la informacion compartida)
#     * Sklearn (para la generacion de los diferentes modelos de KNN Vecinos, utilizando sus siguientes dos modulos:)
#         * KNeighborsRegressor (crea modelos de regresion KNN, lo que nos permitira generar predicciones de la demanda de bicicletas)
#         * mean_squared_error (metrica de medicion de los errores de la prediccion, en contraste con los datos reales)


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error




# ### 1.2 Lectura de la base de datos
# Leemos la base de datos proporcionada, utilizando la libreria de pandas.
# Dentro de la funcion de read_csv hemos definido el parametro de "encoding" como "latin1" para que pueda leer caracteres especiales, ya sea en los nombres de las variables y/o en los valores de estas

bd_bikepro=pd.read_csv(r"C:\Users\cjcamper.AFIRME\Documents\Certificacion AI Core\Primer_reto\Recursos\Data\SeoulBikeData.csv",encoding='latin1')

# Hacemos un head para visualizar las primeras filas de nuestra base

bd_bikepro.head()



# ### 1.3 Modificacion de nombres y valores de variables


# #### Ajustes a realizar a los nombres de nuestras columnas:
# Realizamos un "one-liner" (codigo en una sola linea) para este bucle de iteracion sobre las columnas y darles un mejor formato, partiendo por los siguientes pasos:
 
# 1. Primero vamos a iterar sobre cada columna del dataset, la cual llamaremos **"col"**
# 2. Luego vamos a convertir cada columna en minusculas con la funcion **lower**
# 3. Al ver los nombres de variables como **"Visibility"**, cuyos nombres estan separados del primer parentesis (en donde tenemos sus respectivas escalas), sustitumos ese parentesis por un parentesis sin espacio con la funcion **replace**
# 4. Despues realizamos un **split** entre los parentesis, que nos separara los nombres de las columnas de los parentesis (ya sin espacio) , para solo quedarnos con el nombre de la variable sin su escala
# 5. En el siguiente paso separamos cada palabra del nombre de la variable, nuevamente usando la funcion **split**, ahora separando por espacios
# 6. Finalmente, con el metodo **join**, juntaremos con un guion bajo como delimitador cada una de las palabras



#las nuevas variables se guardaran en el elemento "new_columns"
new_columns=["_".join(col.lower().replace(' (','(').split('(')[0].split(' ')) for col in bd_bikepro.columns]




#ahora asignamos el elemento "new_columns" a nuestra base de datos
bd_bikepro.columns=new_columns


# ### Modificacion de formato para los valores de columnas
# La variable "date" juega un rol fundamental ya que nos permite ordenar nuestros datos, por lo que necesitamos que esta en el formato correcto de fecha



#convertimos en formato fecha la variable de "date"
bd_bikepro['date'] = pd.to_datetime(bd_bikepro['date'], format='%d/%m/%Y')


# ### 1.4 Ordenamiento de datos
# Tal como se comento, ordenaremos nuestra informacion por fecha y hora. Este paso es muy importante ya que necesitamos considerar toda la tendencia historica que ha habido en la demanda de bicicletas, ya que este dato sera vital para la construccion del modelo y la prediccion que nos dara

# La funcion "sort_values", definiendo los campos para ordenacion, nos ayudara a ordenar nuestros datos, tomando el parametro "inplace" como **True** para que realiza la ordenacion en nuestro dataset


bd_bikepro.sort_values(by=['date','hour'],inplace=True)


# ### 1.5 Seleccion de variables predictora y variable objetivo
# Antes de generar nuestro conjunto de entrenamiento y prueba, es importante definir cuales variables tomaremos en cuenta para la construccion de nuestro modelo, asi≠ como la variable que intentaremos predecir (cantidad de bicicletas rentadas)

# Partiremos generando una lista donde contendremos las variables relacionadas al clima, la cual llamaremos "weather_cols"


#Seleccionaremos nuestras columnas del clima descartando todas aquellas variables que no estan relacionadas al clima
weather_cols=[col for col in bd_bikepro.columns if col not in ['date','rented_bike_count','hour','seasons','holiday','functioning_day']]



#generamos una lista con nuestra unica variable a predecir
target_col = ['rented_bike_count']


# ### 1.6 Creacion de conjunto de entrenamiento y prueba

# Para la ejecucion de los modelos de aprendizaje automatico es importante generar dos conjuntos, uno de entrenamiento y otro de prueba, ya sea para las variables que predeciran los valores, asi como la variable objetivo a predecir, respectivamente

# * En el conjunto de **entrenamiento** construiremos nuestro modelo de KNN
# * En el conjunto de **prueba** buscamos comparar los resultados que se predijeron contra los resultados reales

# La demanda de bicicletas tiene un componente temporal, es decir, la demanda ha sido influida por la hora y d√≠a en la que se rentaron las bicicletas, por lo que es importante que nuestros conjuntos de entrenamiento y prueba sigan captando las tendencias mostradas. Debido a lo anterior, generaremos nuestros conjuntos dividiendo el dataset original desde determinado registro



# Datos de entrenamiento
x_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][weather_cols]
y_train = bd_bikepro.loc[: bd_bikepro.shape[0]-1440,:][target_col]

# Datos de test
x_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][weather_cols]
y_test = bd_bikepro.loc[bd_bikepro.shape[0]-1440+1:,:][target_col]



# ### 1.7 Generacion de modelos de KNN por diferentes valores de K Vecinos
# La cantidad de K Vecinos que consideramos para nuestro modelo juega un papel importante a la hora de decidir los mejores parametros de nuestro modelo, buscando tener la menor diferencia entre los valores que se predicen contra los valores reales

# Por lo que generaremos una lista con diferentes valores de K, con el objetivo de iterar entre cada elemento y probar el modelo


# **Objetivo**: Lograr encontrar el valor de K cuyo error sea el menor entre los valores a predecir contra los valores reales




#definimos nuestra lista de K, en el que iteraremos para encontrar el mejor par√°metro para nuestro modelo
vect_k=[3,5,10,15,20,50,100,300,500,1000]


#     1. Generaremos un ciclo en el que iteraremos en cada elemento de nuestra lista
#     2. Tomremos cada elemento de nuestra lista para considerarlo como la cantidad de vecinos a considerar en el modelo
#     3. Se construira el modelo
#     4. Se generaran las predicciones a partir de nuestro modelo, tomando **el conjunto de prueba**
#     5. Se medira el error entre las predicciones del modelo contra los datos reales, tomando el error cuadratico medio
#     6. Se estara acumulando cada error en una lista
#     7. Se estara acumulando el i≠ndice de precision (accuracy), considerando nuestros conjuntos de prueba




#definimos un vector donde estaremos depositando nuestros errores
vect_error=[]
#definimos un vector donde depositaremos el accuracy entre cada iteracion de k
vect_score=[]


for i in range(len(vect_k)):
    model = KNeighborsRegressor(n_neighbors=vect_k[i])
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    error=mean_squared_error(y_test,y_test_pred)
    vect_error.append(error)
    vect_score.append(model.score(x_test,y_test))


# # 1.8 Medicion de nuestros modelos por diferentes valores de K
# Generaremos una grafica en la que podamos visualizar todos los errores correspondientes a cada modelo, por lo que nos apoyaremos de la libreria matplotlib




#el atributo "facecolor" permitira que, cuando queramos exportar nuestra grafica, puedan observarse los titulos de los ejes, asi≠ como sus escalas
plt.figure(figsize=(25,5),facecolor=("white"))

plt.plot(vect_k,vect_error)
plt.title("Grafico Comparativo de error entre elementos de K")
plt.xlabel("K")
plt.ylabel("MSE")
#asimismo, con la funcion "axvline", en el punto 20 de nuestro eje x, dibujaremos una li≠nea recta, donde se observa el menor error posible entre todos los modelos
plt.axvline(x=20,color='red',linestyle='--')

plt.savefig('error_knn_weather.png',orientation ='landscape')


# Hacemos un nuevo grafico, ahora solo quedandonos con nuestros elementos hasta 50, esto solo para visualizar mejor el valor del error en el modelo de 20




#hacemos un nuevo grsfico haciendo zoom hasta el elemento 50 de los k vecinos
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(50,12),facecolor=("white"))


plt.plot(vect_k[:6],vect_error[:6])
plt.title("Grafico Comparativo de error entre elementos de K (hasta 50)",color="black")
plt.xlabel("K")
plt.ylabel("MSE")
plt.axvline(x=20,color='red',linestyle='--')

#asimismo, incluimos los valores de x y y (los valores de los K Vecinos y el valor del error, respectivamente)
#utilizamos la funcion "zip" para imprmir ambos valores, como coordenadas
for x,y in zip(vect_k[:6],vect_error[:6]):
    plt.text(x,y,s=(x,'{:,.2f}'.format(y)),fontsize=15)
#en la impresi√≥n del valor del error ponemos formato a dos decimales
    
plt.savefig('error_knn_weather_hasta_50.png',orientation='landscape',bbox_inches='tight',transparent=False)
plt.show()
#de esta manera se observa mejor como en el valor 20 del vector de vecinos se obtiene el menor valor de error, asi≠ como despues de ese valor el error tiende a incrementarse


# Ya por ultimo, considerando 20 como valor optimo a considerar para nuestro modelo KNN, tomamos este valor como par√°metro, buscando visualizar en un grafico los datos reales contra los valores que se predijeron



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





#el error en el conjunto de entrenamiento fue 201694.03053339708
#el error en el conjunto de test fue 318367.1524027103
#se mantiene la tendencia observada a que el error en el conjunto de test no es parecido al error en el conjunto de entrenamiento






