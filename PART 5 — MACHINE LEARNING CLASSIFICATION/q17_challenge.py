import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

df=load_breast_cancer()
print(df.keys())

X=pd.DataFrame(df.data,columns=df.feature_names)
y=df.target
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scale=StandardScaler()
X_train_scale=scale.fit_transform(X_train)
X_test_scale=scale.transform(X_test)

model=LogisticRegression(max_iter=100000)
model.fit(X_train_scale,y_train)
y_pred=model.predict(X_test_scale)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)

print(accuracy,precision,recall)

confusion_matric=confusion_matrix(y_test,y_pred)
print(confusion_matric)

