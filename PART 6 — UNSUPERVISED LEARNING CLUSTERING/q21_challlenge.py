import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)

print(df.keys())
X=df[["sepal length (cm)","sepal width (cm)"]]

scale=StandardScaler()
X_scale=scale.fit_transform(X)

model=KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
model.fit(X_scale)

labels=model.labels_

plt.figure()
plt.scatter(
    X_scale[:,0],
    X_scale[:,1],
    c=labels
)
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    marker="X",
    color="Red",
    s=200,
    label="centroid"
    
)
plt.legend()
plt.show()