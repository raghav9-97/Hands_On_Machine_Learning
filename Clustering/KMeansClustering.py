import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')

X = data.iloc[:, 3:5].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_means = kmeans.fit_predict(X)

plt.scatter(X[y_means==0,0],X[y_means==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=100,c='cyan',label='Cluster 3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=100,c='magenta',label='Cluster 4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=100,c='green',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.legend()
plt.show()
