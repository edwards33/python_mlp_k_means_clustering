from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data[:, 1:3]

model_means = KMeans(n_clusters=5, random_state=0)
model_means.fit(x)

centroids = model_means.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', s=170, zorder=10, c='m')
plt.scatter(x[:, 0], x[:, 1], c=model_means.labels_)
plt.xlabel("Width")
plt.ylabel("Length")
plt.show()
