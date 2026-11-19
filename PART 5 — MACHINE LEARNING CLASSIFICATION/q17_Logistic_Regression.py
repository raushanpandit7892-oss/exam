import pandas as pd 
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
df=load_breast_cancer()
X=pd.DataFrame(df.data,columns=df.feature_names)
y=df.target
print(X.head())
print(df.target_names)

print(X.info())
print(X.describe())
print(X.isnull().sum())
print(pd.Series(y).value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model=LogisticRegression(max_iter=50000)
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)
print("Predicted:", y_pred)
print("Actual:", y_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)