import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()

print(iris.keys())

X=pd.DataFrame(iris.data,columns=iris.feature_names)
y=iris.target

df=X.copy()
df["Species"]=y

print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["Species"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Actual:", y_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)