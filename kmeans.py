# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import ting the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values # взяли почему-то только 3 и 4 параметры??

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11): # [1, 11) от 1 до 10 включительно
  kmeans = KMeans(
      n_clusters = i, # количество кластеров
      init = 'k-means++', # что бы не было ошибки кластеризации рандомной
      max_iter = 300, # максимальное количество итераций финального кластера
      n_init = 10, # количество запусков с различными начальными центроидами
      random_state = 0
    )
  kmeans.fit(X) # закидываем данные
  wcss.append(kmeans.inertia_) # считаем качество модели (вес расстояний)

plt.plot(range(1, 11), wcss) # x = 1 - 10, y = wcss
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() # из графика видно, что оптимальное значение = 5

# Applyin k-means to the mall dataset
kmeans = KMeans(
    n_clusters = 5, # [0,4]
    init = 'k-means++',
    max_iter = 300,
    n_init = 10,
    random_state = 0
  )
# строим резльтаты, какому кластеру [0,4] принадлежит клиент
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(
    X[y_kmeans == 0, 0], # указали условие выборки X для 1 кластера (0)
    X[y_kmeans == 0, 1], # указали условие выборки Y для 1 кластера (0)
    s = 100, # вес кисти
    c = 'red',
    label = 'Careful'
  )
plt.scatter(
    X[y_kmeans == 1, 0], # указали условие выборки X для 2 кластера (1)
    X[y_kmeans == 1, 1], # указали условие выборки Y для 2 кластера (1)
    s = 100, # размер кластера
    c = 'blue',
    label = 'Standard'
  )
plt.scatter(
    X[y_kmeans == 2, 0], # указали условие выборки X для 3 кластера (2)
    X[y_kmeans == 2, 1], # указали условие выборки Y для 3 кластера (2)
    s = 100, # вес кисти
    c = 'green',
    label = 'Target'
  )
plt.scatter(
    X[y_kmeans == 3, 0], # указали условие выборки X для 4 кластера (3)
    X[y_kmeans == 3, 1], # указали условие выборки Y для 4 кластера (3)
    s = 100, # вес кисти
    c = 'cyan',
    label = 'Careless'
  )
plt.scatter(
    X[y_kmeans == 4, 0], # указали условие выборки X для 5 кластера (4)
    X[y_kmeans == 4, 1], # указали условие выборки Y для 5 кластера (4)
    s = 100, # вес кисти
    c = 'magenta',
    label = 'Sensible'
  )
# print Cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s = 300, # вес данного рисунка, насколько жирный он будет !!!
    c = 'yellow',
    label = 'Centroids'
  )
plt.title('Clusters of clients')
plt.xlabel('Annual Incone (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()






















