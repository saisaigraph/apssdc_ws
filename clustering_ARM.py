# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

print(os.getcwd())

os.chdir('F:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')


#import random
#import math

from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')

df = dataset

df.drop(["CustomerID"], axis = 1, inplace=True)

plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
#sns.axes_style("dark")
sns.violinplot(y=df["Age"])
plt.show()


genders = df.Genre.value_counts()
genders

sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.show()


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["Spending Score (1-100)"], color="red")
plt.subplot(1,2,2)
sns.boxplot(y=df["Annual Income (k$)"])
plt.show()









age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age18_25


age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]

y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]
y

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()








dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]].values
X

tmpDF = pd.DataFrame(X)
tmpDF

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)

#X[y_kmeans == 0, 0], X[y_kmeans == 0, 1]

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careless')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Sensible')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Careful')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


dataset["clusters"] = kmeans.labels_

dataset.head()

dataset.sample(5)

centers = pd.DataFrame(kmeans.cluster_centers_)
centers

'''centers["clusters"] = range(5) #n_clusters
centers

dataset["ind"] = dataset.index
dataset.head()

dataset = dataset.merge(centers)
dataset

dataset.sample(20)

dataset = dataset.sort_values("ind")
dataset.head()


dataset = dataset.drop("ind",1)
dataset.head()'''

'''s1_grps = pd.Series(kmeans.labels_)
s2_univs = dataset.iloc[:,0]
rslt = pd.concat([s1_grps,s2_univs],axis=1)
rslt'''

#sns.lmplot("x","y",data = dataset,fit_reg=False,hue="clusters",size=7)



# Using the elbow method to find the num of clusters
wcss = []

for i in range(1,11):
    print(i)

for i in range(2,11):
    print(i)
    
for i in range(1,11):
    print(i)
    km = KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
    print(km.inertia_)

plt.plot(range(1,11),wcss)    
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
