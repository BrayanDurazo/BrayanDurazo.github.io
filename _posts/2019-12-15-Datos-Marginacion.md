---
layout: post
title: Análisis de marginación en México
date: 2019-12-16 13:32:20 +0300
description: Proyecto para la materia de reconocimiento de patrones, en donde aplico lo aprendido con datos ofrecitos por el Consejo Nacional de Población.
img: post-conapo.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [IA,CONAPO]
---

Proyecto para la materia de reconocimiento de patrones, en donde aplico lo aprendido con datos ofrecitos por el Consejo Nacional de Población.

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns ; sns.set()
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import linear_model
import statsmodels.api as sm
```

## Cargando los datos

Para mas información sobre los datos: https://datos.gob.mx/busca/dataset/indice-de-marginacion-carencias-poblacionales-por-localidad-municipio-y-entidad/resource/36698e5c-79e7-4c74-b48c-2edb9719decc

Estos fueron proporcionados por el Consejo Nacional de Población (CONAPO).


```python
indices_marginacion = pd.read_csv("Base_Indice_de_marginacion_municipal_90-15.csv", encoding = "ISO-8859-1", index_col="MUN")
```

## Visualizar de manera muy general los datos
Más que nada para saber la cantidad de datos que estamos manejando


```python
print(indices_marginacion.shape,"\n")
print(list(indices_marginacion.columns))
```

    (14646, 22) 
    
    ['CVE_ENT', 'ENT', 'CVE_MUN', 'POB_TOT', 'VP', 'ANALF', 'SPRIM', 'OVSDE', 'OVSEE', 'OVSAE', 'VHAC', 'OVPT', 'PL<5000', 'PO2SM', 'OVSD', 'OVSDSE', 'IM', 'GM', 'IND0A100', 'LUG_NAC', 'LUGAR_EST', 'AÑO']


A continuación mostramos el diccionario proporcionado por CONAPO, donde explica a que se refieren cada una las claves de las columnas, el diccionario se encuentra en el siguiente link https://github.com/BrayanDurazo/EvidenciasCursoML/blob/master/Dicc_mun.pdf


```python
indices_marginacion.head(15)
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">

  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CVE_ENT</th>
      <th>ENT</th>
      <th>CVE_MUN</th>
      <th>POB_TOT</th>
      <th>VP</th>
      <th>ANALF</th>
      <th>SPRIM</th>
      <th>OVSDE</th>
      <th>OVSEE</th>
      <th>OVSAE</th>
      <th>...</th>
      <th>PL&lt;5000</th>
      <th>PO2SM</th>
      <th>OVSD</th>
      <th>OVSDSE</th>
      <th>IM</th>
      <th>GM</th>
      <th>IND0A100</th>
      <th>LUG_NAC</th>
      <th>LUGAR_EST</th>
      <th>AÑO</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aguascalientes</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1001</td>
      <td>877190</td>
      <td>-</td>
      <td>2.06</td>
      <td>9.54</td>
      <td>0.31</td>
      <td>0.16</td>
      <td>0.72</td>
      <td>...</td>
      <td>8.73</td>
      <td>31.13</td>
      <td>-</td>
      <td>-</td>
      <td>-1.676</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2408</td>
      <td>11</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>Aguascalientes</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1001</td>
      <td>797010</td>
      <td>-</td>
      <td>2.59</td>
      <td>11.48</td>
      <td>0.49</td>
      <td>0.31</td>
      <td>0.77</td>
      <td>...</td>
      <td>8.73</td>
      <td>29.9</td>
      <td>-</td>
      <td>-</td>
      <td>-1.768</td>
      <td>Muy bajo</td>
      <td>7.69</td>
      <td>2409</td>
      <td>-</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>Aguascalientes</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1001</td>
      <td>723043</td>
      <td>-</td>
      <td>3.19</td>
      <td>13.61</td>
      <td>0.77</td>
      <td>0.54</td>
      <td>1.54</td>
      <td>...</td>
      <td>8.21</td>
      <td>28.37</td>
      <td>-</td>
      <td>-</td>
      <td>-1.831</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2419</td>
      <td>11</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>Aguascalientes</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1001</td>
      <td>643419</td>
      <td>-</td>
      <td>3.86</td>
      <td>18.04</td>
      <td>-</td>
      <td>1.12</td>
      <td>0.88</td>
      <td>...</td>
      <td>7.67</td>
      <td>37.24</td>
      <td>-</td>
      <td>1.5</td>
      <td>-1.871</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2408</td>
      <td>11</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>Aguascalientes</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1001</td>
      <td>582827</td>
      <td>121790</td>
      <td>4.53</td>
      <td>-</td>
      <td>-</td>
      <td>1.62</td>
      <td>1.14</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>2.12</td>
      <td>-</td>
      <td>-1.735</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2393</td>
      <td>-</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>Aguascalientes</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1001</td>
      <td>506274</td>
      <td>-</td>
      <td>6.05</td>
      <td>27.99</td>
      <td>6.55</td>
      <td>3.64</td>
      <td>3</td>
      <td>...</td>
      <td>11.47</td>
      <td>58.36</td>
      <td>-</td>
      <td>-</td>
      <td>-1.833</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2341</td>
      <td>9</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>Jesús María</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1005</td>
      <td>120405</td>
      <td>-</td>
      <td>3.26</td>
      <td>13.73</td>
      <td>0.44</td>
      <td>0.37</td>
      <td>0.73</td>
      <td>...</td>
      <td>45.17</td>
      <td>33.77</td>
      <td>-</td>
      <td>-</td>
      <td>-1.256</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2229</td>
      <td>10</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>Jesús María</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1005</td>
      <td>99590</td>
      <td>-</td>
      <td>3.76</td>
      <td>18.21</td>
      <td>1.26</td>
      <td>1.17</td>
      <td>1.43</td>
      <td>...</td>
      <td>45.17</td>
      <td>31.15</td>
      <td>-</td>
      <td>-</td>
      <td>-1.262</td>
      <td>Muy bajo</td>
      <td>13.411</td>
      <td>2202</td>
      <td>-</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>Jesús María</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1005</td>
      <td>82623</td>
      <td>-</td>
      <td>5.33</td>
      <td>23.64</td>
      <td>1.57</td>
      <td>1.4</td>
      <td>2.48</td>
      <td>...</td>
      <td>41.58</td>
      <td>36.01</td>
      <td>-</td>
      <td>-</td>
      <td>-1.234</td>
      <td>Muy bajo</td>
      <td>-</td>
      <td>2188</td>
      <td>9</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>Jesús María</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1005</td>
      <td>64097</td>
      <td>-</td>
      <td>6.63</td>
      <td>32.89</td>
      <td>-</td>
      <td>2.72</td>
      <td>1.73</td>
      <td>...</td>
      <td>42.28</td>
      <td>47.27</td>
      <td>-</td>
      <td>3.83</td>
      <td>-1.141</td>
      <td>Bajo</td>
      <td>-</td>
      <td>2104</td>
      <td>9</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>Calvillo</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1003</td>
      <td>56048</td>
      <td>-</td>
      <td>4.8</td>
      <td>24.18</td>
      <td>0.55</td>
      <td>0.41</td>
      <td>0.86</td>
      <td>...</td>
      <td>50.76</td>
      <td>61.95</td>
      <td>-</td>
      <td>-</td>
      <td>-0.698</td>
      <td>Bajo</td>
      <td>-</td>
      <td>1799</td>
      <td>5</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>Jesús María</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1005</td>
      <td>54476</td>
      <td>9660</td>
      <td>8.3</td>
      <td>-</td>
      <td>-</td>
      <td>5.57</td>
      <td>3.11</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>7.75</td>
      <td>-</td>
      <td>-0.972</td>
      <td>Bajo</td>
      <td>-</td>
      <td>1975</td>
      <td>-</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>Calvillo</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1003</td>
      <td>54136</td>
      <td>-</td>
      <td>6.19</td>
      <td>28.88</td>
      <td>0.79</td>
      <td>0.87</td>
      <td>1.8</td>
      <td>...</td>
      <td>50.76</td>
      <td>60.08</td>
      <td>-</td>
      <td>-</td>
      <td>-0.754</td>
      <td>Bajo</td>
      <td>19.152</td>
      <td>1836</td>
      <td>-</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>Rincón de Romos</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1007</td>
      <td>53866</td>
      <td>-</td>
      <td>3.53</td>
      <td>14.75</td>
      <td>1.97</td>
      <td>0.52</td>
      <td>1.52</td>
      <td>...</td>
      <td>43.06</td>
      <td>43.44</td>
      <td>-</td>
      <td>-</td>
      <td>-1.045</td>
      <td>Bajo</td>
      <td>-</td>
      <td>2090</td>
      <td>7</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>Calvillo</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>1003</td>
      <td>51658</td>
      <td>9537</td>
      <td>7.49</td>
      <td>-</td>
      <td>-</td>
      <td>3.49</td>
      <td>4.25</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>8.4</td>
      <td>-</td>
      <td>-1.105</td>
      <td>Bajo</td>
      <td>-</td>
      <td>2057</td>
      <td>-</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 22 columns</p>
</div>



### Comenzamos a manejar la información
Como primer paso, modificaremos el tipo de datos en que se maneja cada columna a uno más adecuado para su estudio y asi revisar cuantos datos nos faltan.


```python
colsNoFloat = ['ENT', 'CVE_ENT', 'CVE_MUN', 'LUGAR_EST', 'LUG_NAC', 'IND0A100', 'AÑO', 'GM', 'POB_TOT']
colsFloat = list(set(indices_marginacion.columns) - set(colsNoFloat))
for cols in colsFloat:
    indices_marginacion[cols] = pd.to_numeric(indices_marginacion[cols], downcast='float', errors='coerce')

colsInt = ['LUGAR_EST', 'LUG_NAC', 'IND0A100', 'AÑO', 'POB_TOT']
for cols in colsInt:
    indices_marginacion[cols] = pd.to_numeric(indices_marginacion[cols], downcast='integer', errors='coerce')

indices_marginacion.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 14646 entries, Aguascalientes to -
    Data columns (total 22 columns):
    CVE_ENT      14646 non-null object
    ENT          14646 non-null object
    CVE_MUN      14646 non-null object
    POB_TOT      14646 non-null int32
    VP           2428 non-null float32
    ANALF        14645 non-null float32
    SPRIM        12217 non-null float32
    OVSDE        9774 non-null float32
    OVSEE        14645 non-null float32
    OVSAE        14645 non-null float32
    VHAC         12217 non-null float32
    OVPT         12217 non-null float32
    PL<5000      12218 non-null float32
    PO2SM        12217 non-null float32
    OVSD         2428 non-null float32
    OVSDSE       2443 non-null float32
    IM           14640 non-null float32
    GM           14646 non-null object
    IND0A100     2456 non-null float64
    LUG_NAC      14640 non-null float64
    LUGAR_EST    9756 non-null float64
    AÑO          14646 non-null int16
    dtypes: float32(13), float64(3), int16(1), int32(1), object(4)
    memory usage: 1.7+ MB


Con los resultados anteriores visualizamos los datos que tenemos y eliminamos aquellas columnas que no nos brindan información reelevante o que carecen de datos.


```python
indices_marginacion = indices_marginacion.drop(columns=['LUG_NAC','LUGAR_EST','GM', 'VP', 'IND0A100'])
indices_marginacion = indices_marginacion.drop(columns=['CVE_ENT', 'CVE_MUN','OVSD', 'OVSDSE', 'OVSDE'])
indices_marginacion.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 14646 entries, Aguascalientes to -
    Data columns (total 12 columns):
    ENT        14646 non-null object
    POB_TOT    14646 non-null int32
    ANALF      14645 non-null float32
    SPRIM      12217 non-null float32
    OVSEE      14645 non-null float32
    OVSAE      14645 non-null float32
    VHAC       12217 non-null float32
    OVPT       12217 non-null float32
    PL<5000    12218 non-null float32
    PO2SM      12217 non-null float32
    IM         14640 non-null float32
    AÑO        14646 non-null int16
    dtypes: float32(9), int16(1), int32(1), object(1)
    memory usage: 829.6+ KB


Realizamos imputación de datos, reemplazando todos aquellos que no tenían un valor, con el promedio de toda su columna.


```python
#Cambio de nombre a columna para poder manejarla con mayor facilidad.
indices_marginacion = indices_marginacion.rename(columns={'PL<5000': 'P_5000'})

indices_marginacion['ANALF'] = indices_marginacion.ANALF.fillna(indices_marginacion['ANALF'].mean())
indices_marginacion['SPRIM'] = indices_marginacion.SPRIM.fillna(indices_marginacion['SPRIM'].mean())
indices_marginacion['OVSEE'] = indices_marginacion.OVSEE.fillna(indices_marginacion['OVSEE'].mean())
indices_marginacion['OVSAE'] = indices_marginacion.OVSAE.fillna(indices_marginacion['OVSAE'].mean())
indices_marginacion['VHAC'] = indices_marginacion.VHAC.fillna(indices_marginacion['VHAC'].mean())
indices_marginacion['OVPT'] = indices_marginacion.OVPT.fillna(indices_marginacion['OVPT'].mean())
indices_marginacion['P_5000'] = indices_marginacion.P_5000.fillna(indices_marginacion['P_5000'].mean())
indices_marginacion['PO2SM'] = indices_marginacion.PO2SM.fillna(indices_marginacion['PO2SM'].mean())
indices_marginacion['IM'] = indices_marginacion.IM.fillna(indices_marginacion['IM'].mean())

indices_marginacion.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 14646 entries, Aguascalientes to -
    Data columns (total 12 columns):
    ENT        14646 non-null object
    POB_TOT    14646 non-null int32
    ANALF      14646 non-null float32
    SPRIM      14646 non-null float32
    OVSEE      14646 non-null float32
    OVSAE      14646 non-null float32
    VHAC       14646 non-null float32
    OVPT       14646 non-null float32
    P_5000     14646 non-null float32
    PO2SM      14646 non-null float32
    IM         14646 non-null float32
    AÑO        14646 non-null int16
    dtypes: float32(9), int16(1), int32(1), object(1)
    memory usage: 829.6+ KB


### Visualizar los datos
A continuación, manejaremos la información para generar gráficos, tratando de entender mejor los datos que tenemos. Como primer vista, agrupando los datos en base a su estado y año, buscamos mostrar el analfabetismo en el país en el año 2015 por estado.


```python
grupo_entidad_año = indices_marginacion.groupby(['ENT','AÑO'])
avrANALF2015 = []

for entidad in indices_marginacion.ENT.unique():
    # Se obitenen los promedios de la columna ANALF por entidad federativa con datos del año 2015
    avrANALF2015.append(grupo_entidad_año.get_group((entidad,2015))['ANALF'].mean())

# Se genera un data frame para relacionar cada entidad con su promedio sobre la columna ANALF
ANALF_2015 = pd.DataFrame({'Entidad':indices_marginacion.ENT.unique(), 'PorcentajePromedio':avrANALF2015})

ANALF_2015 = ANALF_2015.sort_values(by='PorcentajePromedio', ascending=False)

# Graficamos
sns.set(font_scale=1.8)
plt.figure(figsize=(23,20))
plt.title('Analfabetismo por estado en México en el año 2015')
ANALF_entidades = sns.barplot(x = ANALF_2015.PorcentajePromedio, y = ANALF_2015.Entidad)
for p in ANALF_entidades.patches:
        x, y = p.get_x() + p.get_width() + 0.02 , p.get_y() + p.get_height() - 0.2
        ANALF_entidades.annotate('{:.1f}%'.format(p.get_width()), (x, y))
```


![image](/assets/img/output_14_0.jpg)



De igual manera se pueden obtener por municipio de cada entidad federativa.


```python
sonora2015 = grupo_entidad_año.get_group(('Sonora', 2015)).sort_values(by='ANALF',ascending=False)
plt.figure(figsize=(22,44))
ANALF_sonora = sns.barplot(x = sonora2015['ANALF'], y = sonora2015.index)
for p in ANALF_sonora.patches:
        x, y = p.get_x() + p.get_width() + 0.02 , p.get_y() + p.get_height() - 0.2
        ANALF_sonora.annotate('{:.1f}%'.format(p.get_width()), (x, y))
```

![image](/assets/img/output_16_0.jpg)




## Entendiendo los datos
Tratamos de etiquetar los datos en base a su indice de marginación.


```python
indices_marginacion = indices_marginacion.drop(["-"])
```


```python
X = indices_marginacion.drop(columns=['ENT','POB_TOT','AÑO','IM', 'P_5000'])
X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 14641 entries, Aguascalientes to El Plateado de Joaquín Amaro
    Data columns (total 7 columns):
    ANALF    14641 non-null float32
    SPRIM    14641 non-null float32
    OVSEE    14641 non-null float32
    OVSAE    14641 non-null float32
    VHAC     14641 non-null float32
    OVPT     14641 non-null float32
    PO2SM    14641 non-null float32
    dtypes: float32(7)
    memory usage: 514.7+ KB


A continuación hacemos uso del método del codo para determinar el número de clusters óptimo para un análisis de k-medias.


```python
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(16,6))
plt.plot(range(1,11),wcss)
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()
```

![image](/assets/img/output_21_0.jpg)




```python
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)
```

Implementamos el análisis de las k-medias y utilizamos el algoritmo PCA reducir la dimension de los datos para poder graficar.


```python
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

Centroids_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(16,6))
X_new = pca.inverse_transform(X_pca)
sns.scatterplot(X_pca[:, 0], X_pca[:, 1], hue=y_kmeans, palette=sns.color_palette("cubehelix", 3))
sns.scatterplot(Centroids_pca[:,0],Centroids_pca[:,1],s=300,label='Centroids')
```

    original shape:    (14641, 7)
    transformed shape: (14641, 2)





    <matplotlib.axes._subplots.AxesSubplot at 0x1c24e46550>



![image](/assets/img/output_24_2.jpg)



Utilizamos pca nuevamente para localizar a los datos más representativos de cada grupo (que tienen menor distancia a los centroides).


```python
pca2 = PCA(n_components=1)
pca2.fit(X_pca)
X_pca2 = pca2.transform(X_pca)
print("original shape:   ", X_pca.shape)
print("transformed shape:", X_pca2.shape)
Centroids_pca2 = pca2.transform(Centroids_pca)

X_prueba = indices_marginacion.copy()
X_prueba["pca"] = X_pca2
X_prueba["Grupo"] = y_kmeans
grupo_año = X_prueba.groupby(['AÑO'])
X_prueba = grupo_año.get_group(2015)

X2015_grupo0 = X_prueba.iloc[(X_prueba.pca-Centroids_pca2[0]).abs().argsort()]
X2015_grupo1 = X_prueba.iloc[(X_prueba.pca-Centroids_pca2[1]).abs().argsort()]
X2015_grupo2 = X_prueba.iloc[(X_prueba.pca-Centroids_pca2[2]).abs().argsort()]
```

    original shape:    (14641, 2)
    transformed shape: (14641, 1)


5 datos más representativos del grupo 0


```python
X2015_grupo0.drop(columns=['pca']).head(5)
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT</th>
      <th>POB_TOT</th>
      <th>ANALF</th>
      <th>SPRIM</th>
      <th>OVSEE</th>
      <th>OVSAE</th>
      <th>VHAC</th>
      <th>OVPT</th>
      <th>P_5000</th>
      <th>PO2SM</th>
      <th>IM</th>
      <th>AÑO</th>
      <th>Grupo</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chiconcuautla</th>
      <td>Puebla</td>
      <td>16569</td>
      <td>32.770000</td>
      <td>49.599998</td>
      <td>0.49</td>
      <td>1.570000</td>
      <td>60.610001</td>
      <td>22.160000</td>
      <td>100.0</td>
      <td>75.699997</td>
      <td>1.581</td>
      <td>2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Yahualica</th>
      <td>Hidalgo</td>
      <td>24173</td>
      <td>31.080000</td>
      <td>43.919998</td>
      <td>2.25</td>
      <td>41.130001</td>
      <td>46.389999</td>
      <td>5.690000</td>
      <td>100.0</td>
      <td>69.970001</td>
      <td>1.367</td>
      <td>2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>San Juan Coatzóspam</th>
      <td>Oaxaca</td>
      <td>1808</td>
      <td>30.360001</td>
      <td>51.970001</td>
      <td>3.96</td>
      <td>11.780000</td>
      <td>50.200001</td>
      <td>7.590000</td>
      <td>100.0</td>
      <td>90.330002</td>
      <td>1.516</td>
      <td>2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>San Miguel Mixtepec</th>
      <td>Oaxaca</td>
      <td>2644</td>
      <td>29.990000</td>
      <td>48.580002</td>
      <td>2.47</td>
      <td>1.750000</td>
      <td>59.000000</td>
      <td>27.290001</td>
      <td>100.0</td>
      <td>71.180000</td>
      <td>1.606</td>
      <td>2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>San José del Peñasco</th>
      <td>Oaxaca</td>
      <td>2035</td>
      <td>9.140000</td>
      <td>34.009998</td>
      <td>5.32</td>
      <td>19.000000</td>
      <td>48.709999</td>
      <td>50.610001</td>
      <td>100.0</td>
      <td>53.520000</td>
      <td>1.277</td>
      <td>2015</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



5 datos más representativos del grupo 1


```python
X2015_grupo1.drop(columns=['pca']).head(5)
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT</th>
      <th>POB_TOT</th>
      <th>ANALF</th>
      <th>SPRIM</th>
      <th>OVSEE</th>
      <th>OVSAE</th>
      <th>VHAC</th>
      <th>OVPT</th>
      <th>P_5000</th>
      <th>PO2SM</th>
      <th>IM</th>
      <th>AÑO</th>
      <th>Grupo</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cuautepec de Hinojosa</th>
      <td>Hidalgo</td>
      <td>58301</td>
      <td>9.80</td>
      <td>24.230000</td>
      <td>1.03</td>
      <td>3.87</td>
      <td>36.049999</td>
      <td>2.25</td>
      <td>65.440002</td>
      <td>56.759998</td>
      <td>-0.299</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Tepetlixpa</th>
      <td>México</td>
      <td>19843</td>
      <td>4.27</td>
      <td>16.070000</td>
      <td>0.58</td>
      <td>10.97</td>
      <td>36.770000</td>
      <td>3.75</td>
      <td>27.760000</td>
      <td>58.340000</td>
      <td>-0.706</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Aldama</th>
      <td>Tamaulipas</td>
      <td>29183</td>
      <td>7.35</td>
      <td>28.420000</td>
      <td>2.24</td>
      <td>6.40</td>
      <td>28.469999</td>
      <td>3.02</td>
      <td>53.639999</td>
      <td>55.919998</td>
      <td>-0.479</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Cadereyta de Montes</th>
      <td>Querétaro de Arteaga</td>
      <td>69549</td>
      <td>10.14</td>
      <td>23.610001</td>
      <td>2.80</td>
      <td>5.56</td>
      <td>41.130001</td>
      <td>2.81</td>
      <td>79.199997</td>
      <td>48.070000</td>
      <td>-0.007</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Santiago Laollaga</th>
      <td>Oaxaca</td>
      <td>3326</td>
      <td>10.98</td>
      <td>29.350000</td>
      <td>3.29</td>
      <td>2.20</td>
      <td>27.830000</td>
      <td>5.81</td>
      <td>100.000000</td>
      <td>53.389999</td>
      <td>-0.143</td>
      <td>2015</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



5 datos más representativos del grupo 2


```python
X2015_grupo2.drop(columns=['pca']).head(5)
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT</th>
      <th>POB_TOT</th>
      <th>ANALF</th>
      <th>SPRIM</th>
      <th>OVSEE</th>
      <th>OVSAE</th>
      <th>VHAC</th>
      <th>OVPT</th>
      <th>P_5000</th>
      <th>PO2SM</th>
      <th>IM</th>
      <th>AÑO</th>
      <th>Grupo</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>San José Tenango</th>
      <td>Oaxaca</td>
      <td>18316</td>
      <td>39.070000</td>
      <td>57.080002</td>
      <td>16.020000</td>
      <td>80.620003</td>
      <td>58.610001</td>
      <td>52.270000</td>
      <td>100.0</td>
      <td>74.750000</td>
      <td>3.808</td>
      <td>2015</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Tehuipango</th>
      <td>Veracruz de Ignacio de la Llave</td>
      <td>26322</td>
      <td>46.869999</td>
      <td>56.119999</td>
      <td>3.430000</td>
      <td>63.150002</td>
      <td>78.459999</td>
      <td>46.279999</td>
      <td>100.0</td>
      <td>73.680000</td>
      <td>3.545</td>
      <td>2015</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Santa María Chilchotla</th>
      <td>Oaxaca</td>
      <td>20328</td>
      <td>32.799999</td>
      <td>54.169998</td>
      <td>11.570000</td>
      <td>66.870003</td>
      <td>54.029999</td>
      <td>43.790001</td>
      <td>100.0</td>
      <td>82.739998</td>
      <td>3.130</td>
      <td>2015</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Mezquital</th>
      <td>Durango</td>
      <td>39288</td>
      <td>26.139999</td>
      <td>43.389999</td>
      <td>57.959999</td>
      <td>56.540001</td>
      <td>61.910000</td>
      <td>55.939999</td>
      <td>100.0</td>
      <td>44.450001</td>
      <td>4.845</td>
      <td>2015</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Cochoapa el Grande</th>
      <td>Guerrero</td>
      <td>18458</td>
      <td>56.419998</td>
      <td>71.239998</td>
      <td>14.590000</td>
      <td>30.549999</td>
      <td>71.440002</td>
      <td>37.619999</td>
      <td>100.0</td>
      <td>76.860001</td>
      <td>4.744</td>
      <td>2015</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Visualicemos los grupos en un mapa


```python
import datetime
import pandas as pd
import geopandas as gpd
import json
from bokeh.io import curdoc, output_notebook
from bokeh.models import Slider, HoverTool
from IPython.display import YouTubeVideo
```

Utilizamos los datos proporcionados en https://www.pakin.lat/datos-geoespaciales-python.html, donde nos proporcionan un geodataframe con el campo de las figuras de cada localidad.


```python
# Declaramos una variable con la ruta al archivo .shp
path = '/Users/brayandurazo/Desktop/EvidenciasCursoML/PHLITL_2000/PHLITL_2000.shp'

# Importamos los datos que nos sirven
data_f = gpd.read_file(path)[['EDO_LEY','MPO_LEY', 'geometry']]
data_f.drop_duplicates('MPO_LEY', inplace = True)
data_f["Grupo"] = 1
data_f["IM"] = -0.3
print(data_f.info())
data_f.head()
```

    <class 'geopandas.geodataframe.GeoDataFrame'>
    Int64Index: 2302 entries, 0 to 2479
    Data columns (total 5 columns):
    EDO_LEY     2302 non-null object
    MPO_LEY     2302 non-null object
    geometry    2302 non-null geometry
    Grupo       2302 non-null int64
    IM          2302 non-null float64
    dtypes: float64(1), geometry(1), int64(1), object(2)
    memory usage: 107.9+ KB
    None





<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EDO_LEY</th>
      <th>MPO_LEY</th>
      <th>geometry</th>
      <th>Grupo</th>
      <th>IM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Baja California</td>
      <td>Mexicali</td>
      <td>POLYGON ((788992.7599999954 3707437.079667801,...</td>
      <td>1</td>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sonora</td>
      <td>San Luis Río Colorado</td>
      <td>POLYGON ((788992.7599999954 3707437.079667801,...</td>
      <td>1</td>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Baja California</td>
      <td>Ensenada</td>
      <td>POLYGON ((783083.2999999949 3526355.799667791,...</td>
      <td>1</td>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sonora</td>
      <td>Puerto Peñasco</td>
      <td>POLYGON ((901967.8399999959 3655406.799667796,...</td>
      <td>1</td>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sonora</td>
      <td>General Plutarco Elías Calle</td>
      <td>POLYGON ((1014518.909999996 3604588.009667793,...</td>
      <td>1</td>
      <td>-0.3</td>
    </tr>
  </tbody>
</table>
</div>



A los datos que obtuvimos con el clustering realizado anteriormente, quitamos columnas que no nos interesan para la presentación en el mapa.


```python
datos_pruebas = X_prueba.drop(columns=['pca','POB_TOT','ANALF','SPRIM','OVSEE','OVSAE','VHAC','OVPT','P_5000','PO2SM','AÑO'])
datos_pruebas['Municipios'] = datos_pruebas.index
datos_pruebas.drop_duplicates('Municipios', inplace = True)
print(datos_pruebas.info())
datos_pruebas.head(5) 
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2317 entries, Aguascalientes to El Plateado de Joaquín Amaro
    Data columns (total 4 columns):
    ENT           2317 non-null object
    IM            2317 non-null float32
    Grupo         2317 non-null int32
    Municipios    2317 non-null object
    dtypes: float32(1), int32(1), object(2)
    memory usage: 72.4+ KB
    None





<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT</th>
      <th>IM</th>
      <th>Grupo</th>
      <th>Municipios</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aguascalientes</th>
      <td>Aguascalientes</td>
      <td>-1.676</td>
      <td>1</td>
      <td>Aguascalientes</td>
    </tr>
    <tr>
      <th>Jesús María</th>
      <td>Aguascalientes</td>
      <td>-1.256</td>
      <td>1</td>
      <td>Jesús María</td>
    </tr>
    <tr>
      <th>Calvillo</th>
      <td>Aguascalientes</td>
      <td>-0.698</td>
      <td>1</td>
      <td>Calvillo</td>
    </tr>
    <tr>
      <th>Rincón de Romos</th>
      <td>Aguascalientes</td>
      <td>-1.045</td>
      <td>1</td>
      <td>Rincón de Romos</td>
    </tr>
    <tr>
      <th>Pabellón de Arteaga</th>
      <td>Aguascalientes</td>
      <td>-1.129</td>
      <td>1</td>
      <td>Pabellón de Arteaga</td>
    </tr>
  </tbody>
</table>
</div>



En el sigueinte recuadro agregamos las columnas de "Grupo" y "IM" al geodataframe.


```python
municipios_faltantes = set()
for municipio in data_f['MPO_LEY']:
    municipios_faltantes.add(municipio)
for municipio in datos_pruebas['Municipios']:
    if(municipio in municipios_faltantes):
        x = datos_pruebas.loc[datos_pruebas.index == municipio]["Grupo"]
        y = datos_pruebas.loc[datos_pruebas.index == municipio]["IM"]
        data_f.at[data_f.index[data_f['MPO_LEY'] == municipio], 'Grupo'] = x[0]
        data_f.at[data_f.index[data_f['MPO_LEY'] == municipio], 'IM'] = y[0]

data_f.head(5)
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EDO_LEY</th>
      <th>MPO_LEY</th>
      <th>geometry</th>
      <th>Grupo</th>
      <th>IM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Baja California</td>
      <td>Mexicali</td>
      <td>POLYGON ((788992.7599999954 3707437.079667801,...</td>
      <td>1</td>
      <td>-1.648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sonora</td>
      <td>San Luis Río Colorado</td>
      <td>POLYGON ((788992.7599999954 3707437.079667801,...</td>
      <td>1</td>
      <td>-1.292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Baja California</td>
      <td>Ensenada</td>
      <td>POLYGON ((783083.2999999949 3526355.799667791,...</td>
      <td>1</td>
      <td>-1.323</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sonora</td>
      <td>Puerto Peñasco</td>
      <td>POLYGON ((901967.8399999959 3655406.799667796,...</td>
      <td>1</td>
      <td>-1.515</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sonora</td>
      <td>General Plutarco Elías Calle</td>
      <td>POLYGON ((1014518.909999996 3604588.009667793,...</td>
      <td>1</td>
      <td>-0.300</td>
    </tr>
  </tbody>
</table>
</div>




```python
m_data = json.loads(data_f.to_json())
json_data = json.dumps(m_data)
```


```python
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer, all_palettes, small_palettes
```


```python
# Creamos un objeto GeoJSON que contiene las variables que graficaremos
geosource = GeoJSONDataSource(geojson = json_data)
```


```python
# Definimos una paleta de colores
palette =  all_palettes['Paired'][3]
#Ordenamos la paleta de colores de forma inversa
palette = palette[::-1]
#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 3)

#Add hover tool
TOOLTIPS = [('ESTADO','@EDO_LEY'),('MUNICIPIO','@MPO_LEY'),('IM','@IM'),('Grupo','@Grupo')]
```


```python
#Create figure object.

p = figure(title = 'Municipios según los clusters',
           plot_height = 300,
           plot_width = 500,
           toolbar_location = 'right',
           tooltips = TOOLTIPS,
           sizing_mode = 'scale_width'
          )


p.xaxis.visible = None
p.yaxis.visible = None
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.title.align = 'center'
p.title.text_font_size = "14px"
```


```python
p.patches('xs','ys', source = geosource, fill_color = {'field':'Grupo','transform':color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
#Configuramos para presentar la imagen inline
output_notebook()

# Muestra la visualización
show(p)
```

![image](/assets/img/output_39_0.jpg)



## Regresión lineal
A continuación, revisaremos si existe una relación lineal entre la variable dependiente (Indice de marginación) y las demás características que tenemos sobre la población


```python
sns.scatterplot(indices_marginacion['ANALF'], indices_marginacion['IM'])
plt.title('Analfabetismo Vs Indice de marginación', fontsize=14)
plt.xlabel('Analfabetismo', fontsize=10)
plt.ylabel('Indice Marginación', fontsize=10)
plt.show()

sns.scatterplot(indices_marginacion['SPRIM'], indices_marginacion['IM'])
plt.title('Primaria incompleta Vs Indice de marginación', fontsize=14)
plt.xlabel('Primaria incompleta', fontsize=10)
plt.ylabel('Indice Marginación', fontsize=10)
plt.show()

sns.scatterplot(indices_marginacion['OVSEE'], indices_marginacion['IM'])
plt.title('Vivienda sin electricidad Vs Indice de marginación', fontsize=14)
plt.xlabel('Vivienda sin electricidad', fontsize=10)
plt.ylabel('Indice Marginación', fontsize=10)
plt.show()

sns.scatterplot(indices_marginacion['OVSAE'], indices_marginacion['IM'])
plt.title('Vivienda sin agua entubada Vs Indice de marginación', fontsize=14)
plt.xlabel('Vivienda sin agua entubada', fontsize=10)
plt.ylabel('Indice Marginación', fontsize=10)
plt.show()

sns.scatterplot(indices_marginacion['VHAC'], indices_marginacion['IM'])
plt.title('Vivienda con hacinamiento Vs Indice de marginación', fontsize=14)
plt.xlabel('Vivienda con hacinamiento', fontsize=10)
plt.ylabel('Indice Marginación', fontsize=10)
plt.show()

sns.scatterplot(indices_marginacion['OVPT'], indices_marginacion['IM'])
plt.title('Vivienda con piso de tierra Vs Indice de marginación', fontsize=14)
plt.xlabel('Vivienda con piso de tierra', fontsize=10)
plt.ylabel('Indice Marginación', fontsize=10)
plt.show()
```

![image](/assets/img/output_50_0.jpg)



![image](/assets/img/output_50_1.jpg)



![image](/assets/img/output_50_2.jpg)



![image](/assets/img/output_50_3.jpg)



![image](/assets/img/output_50_4.jpg)



![image](/assets/img/output_50_5.jpg)



```python
X = indices_marginacion.drop(columns=['ENT','P_5000','AÑO','IM','PO2SM'])
Y = indices_marginacion['IM']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
```

    Intercept: 
     -1.0315167499349738
    Coefficients: 
     [-1.29089668e-06  5.41096983e-02 -8.42236805e-04 -1.01935661e-03
      9.08252723e-03 -1.22710499e-03  3.10081426e-03]



```python
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)
```


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     IM   R-squared:                       0.725
    Model:                            OLS   Adj. R-squared:                  0.725
    Method:                 Least Squares   F-statistic:                     5510.
    Date:                Fri, 13 Dec 2019   Prob (F-statistic):               0.00
    Time:                        08:45:28   Log-Likelihood:                -11322.
    No. Observations:               14641   AIC:                         2.266e+04
    Df Residuals:                   14633   BIC:                         2.272e+04
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -1.0315      0.019    -53.121      0.000      -1.070      -0.993
    POB_TOT    -1.291e-06   3.72e-08    -34.704      0.000   -1.36e-06   -1.22e-06
    ANALF          0.0541      0.001     93.420      0.000       0.053       0.055
    SPRIM         -0.0008      0.001     -1.587      0.113      -0.002       0.000
    OVSEE         -0.0010      0.000     -2.559      0.011      -0.002      -0.000
    OVSAE          0.0091      0.000     33.772      0.000       0.009       0.010
    VHAC          -0.0012      0.000     -2.692      0.007      -0.002      -0.000
    OVPT           0.0031      0.000      9.021      0.000       0.002       0.004
    ==============================================================================
    Omnibus:                      793.150   Durbin-Watson:                   1.631
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1237.777
    Skew:                           0.463   Prob(JB):                    1.66e-269
    Kurtosis:                       4.082   Cond. No.                     5.80e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.8e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.




```python
random_subset = indices_marginacion.sample(10)
random_subset_predict = random_subset.drop(columns=['ENT','P_5000','AÑO','IM','PO2SM'])
random_subset
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT</th>
      <th>POB_TOT</th>
      <th>ANALF</th>
      <th>SPRIM</th>
      <th>OVSEE</th>
      <th>OVSAE</th>
      <th>VHAC</th>
      <th>OVPT</th>
      <th>P_5000</th>
      <th>PO2SM</th>
      <th>IM</th>
      <th>AÑO</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Tecpatán</th>
      <td>Chiapas</td>
      <td>41045</td>
      <td>18.450001</td>
      <td>45.150002</td>
      <td>6.960000</td>
      <td>12.880000</td>
      <td>57.340000</td>
      <td>10.860000</td>
      <td>83.389999</td>
      <td>79.470001</td>
      <td>0.635</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>Huaquechula</th>
      <td>Puebla</td>
      <td>28654</td>
      <td>15.540000</td>
      <td>52.320000</td>
      <td>3.310000</td>
      <td>17.799999</td>
      <td>65.480003</td>
      <td>33.939999</td>
      <td>100.000000</td>
      <td>90.930000</td>
      <td>0.417</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>El Oro</th>
      <td>Durango</td>
      <td>11320</td>
      <td>3.840000</td>
      <td>27.010000</td>
      <td>1.720000</td>
      <td>6.260000</td>
      <td>23.600000</td>
      <td>4.010000</td>
      <td>48.070000</td>
      <td>55.810001</td>
      <td>-0.939</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>Huehuetán</th>
      <td>Chiapas</td>
      <td>28335</td>
      <td>26.250000</td>
      <td>60.500000</td>
      <td>39.490002</td>
      <td>64.519997</td>
      <td>79.839996</td>
      <td>51.189999</td>
      <td>80.550003</td>
      <td>84.360001</td>
      <td>0.690</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>Santa Catarina Lachatao</th>
      <td>Oaxaca</td>
      <td>1558</td>
      <td>6.970000</td>
      <td>41.215691</td>
      <td>4.040000</td>
      <td>10.910000</td>
      <td>50.864685</td>
      <td>23.662128</td>
      <td>73.614311</td>
      <td>66.767021</td>
      <td>0.234</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>Guerrero</th>
      <td>Tamaulipas</td>
      <td>4477</td>
      <td>4.110000</td>
      <td>22.820000</td>
      <td>0.680000</td>
      <td>1.660000</td>
      <td>20.920000</td>
      <td>1.410000</td>
      <td>100.000000</td>
      <td>32.730000</td>
      <td>-1.206</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>Temósachi</th>
      <td>Chihuahua</td>
      <td>9021</td>
      <td>12.640000</td>
      <td>61.299999</td>
      <td>58.169998</td>
      <td>39.700001</td>
      <td>52.180000</td>
      <td>25.670000</td>
      <td>100.000000</td>
      <td>74.260002</td>
      <td>-0.189</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>Cuajinicuilapa</th>
      <td>Guerrero</td>
      <td>27266</td>
      <td>18.030001</td>
      <td>37.580002</td>
      <td>1.850000</td>
      <td>8.970000</td>
      <td>40.529999</td>
      <td>11.010000</td>
      <td>60.340000</td>
      <td>61.340000</td>
      <td>0.575</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>Santa Isabel Cholula</th>
      <td>Puebla</td>
      <td>8188</td>
      <td>13.020000</td>
      <td>41.215691</td>
      <td>2.960000</td>
      <td>28.299999</td>
      <td>50.864685</td>
      <td>23.662128</td>
      <td>73.614311</td>
      <td>66.767021</td>
      <td>0.187</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>San Pedro Ocotepec</th>
      <td>Oaxaca</td>
      <td>2098</td>
      <td>18.040001</td>
      <td>41.840000</td>
      <td>8.800000</td>
      <td>0.860000</td>
      <td>47.189999</td>
      <td>8.890000</td>
      <td>100.000000</td>
      <td>46.509998</td>
      <td>0.651</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = regr.predict(random_subset_predict)
random_subset_predict['IM'] = a
random_subset_predict
```




<div style="overflow:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>POB_TOT</th>
      <th>ANALF</th>
      <th>SPRIM</th>
      <th>OVSEE</th>
      <th>OVSAE</th>
      <th>VHAC</th>
      <th>OVPT</th>
      <th>IM</th>
    </tr>
    <tr>
      <th>MUN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Tecpatán</th>
      <td>41045</td>
      <td>18.450001</td>
      <td>45.150002</td>
      <td>6.960000</td>
      <td>12.880000</td>
      <td>57.340000</td>
      <td>10.860000</td>
      <td>-0.051004</td>
    </tr>
    <tr>
      <th>Huaquechula</th>
      <td>28654</td>
      <td>15.540000</td>
      <td>52.320000</td>
      <td>3.310000</td>
      <td>17.799999</td>
      <td>65.480003</td>
      <td>33.939999</td>
      <td>-0.088522</td>
    </tr>
    <tr>
      <th>El Oro</th>
      <td>11320</td>
      <td>3.840000</td>
      <td>27.010000</td>
      <td>1.720000</td>
      <td>6.260000</td>
      <td>23.600000</td>
      <td>4.010000</td>
      <td>-0.822519</td>
    </tr>
    <tr>
      <th>Huehuetán</th>
      <td>28335</td>
      <td>26.250000</td>
      <td>60.500000</td>
      <td>39.490002</td>
      <td>64.519997</td>
      <td>79.839996</td>
      <td>51.189999</td>
      <td>0.907839</td>
    </tr>
    <tr>
      <th>Santa Catarina Lachatao</th>
      <td>1558</td>
      <td>6.970000</td>
      <td>41.215691</td>
      <td>4.040000</td>
      <td>10.910000</td>
      <td>50.864685</td>
      <td>23.662128</td>
      <td>-0.585169</td>
    </tr>
    <tr>
      <th>Guerrero</th>
      <td>4477</td>
      <td>4.110000</td>
      <td>22.820000</td>
      <td>0.680000</td>
      <td>1.660000</td>
      <td>20.920000</td>
      <td>1.410000</td>
      <td>-0.841040</td>
    </tr>
    <tr>
      <th>Temósachi</th>
      <td>9021</td>
      <td>12.640000</td>
      <td>61.299999</td>
      <td>58.169998</td>
      <td>39.700001</td>
      <td>52.180000</td>
      <td>25.670000</td>
      <td>-0.093997</td>
    </tr>
    <tr>
      <th>Cuajinicuilapa</th>
      <td>27266</td>
      <td>18.030001</td>
      <td>37.580002</td>
      <td>1.850000</td>
      <td>8.970000</td>
      <td>40.529999</td>
      <td>11.010000</td>
      <td>-0.058778</td>
    </tr>
    <tr>
      <th>Santa Isabel Cholula</th>
      <td>8188</td>
      <td>13.020000</td>
      <td>41.215691</td>
      <td>2.960000</td>
      <td>28.299999</td>
      <td>50.864685</td>
      <td>23.662128</td>
      <td>-0.107318</td>
    </tr>
    <tr>
      <th>San Pedro Ocotepec</th>
      <td>2098</td>
      <td>18.040001</td>
      <td>41.840000</td>
      <td>8.800000</td>
      <td>0.860000</td>
      <td>47.189999</td>
      <td>8.890000</td>
      <td>-0.124825</td>
    </tr>
  </tbody>
</table>
</div>

## Conclusiones
Con la información observada podemos concluir múltiples ideas, una que se ve claramente en el mapa que se generó, podemos ver fácilmente que en el sur del país los índices de marginación son mayores que en el norte.

A través de la regresión lineal pudimos observar que el analfabetismo es la característica que mas inflye en el índice de marginación, por lo que la educación a ese nivel debe de ser prioritario para el país. 


```python

```
