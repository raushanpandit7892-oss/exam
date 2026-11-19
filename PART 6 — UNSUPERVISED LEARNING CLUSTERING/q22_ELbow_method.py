import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)

X=df[["sepal length (cm)","sepal width (cm)"]]
print(X.head())
print(X.info())
print(X.describe())
print(X.isnull().sum())

scale=StandardScaler()
X_scale=scale.fit_transform(X)

model=KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
model.fit(X_scale)
labels=model.labels_
print("Cluster Lable: ",labels)
print("Inertia: ",model.inertia_)

plt.figure()
plt.scatter(
    X_scale[:,0],
    X_scale[:,1],
    c=labels,
)
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.legend()
plt.show()

wcss=[]
for i in range(1,11):
    model=KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    model.fit(X_scale)
    wcss.append(model.inertia_)
print(wcss)
plt.plot(range(1,11),wcss,marker="X")
plt.show()