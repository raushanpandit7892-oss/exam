import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
)
df=load_breast_cancer()
X=pd.DataFrame(df.data, columns=df.feature_names)
y=df.target

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

model=SVC(kernel="linear")
model.fit(X_train_scale,y_train)
y_pred=model.predict(X_test_scale)

accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
class_report=classification_report(y_test,y_pred)
print(accuracy)
print(cm)
print(class_report)
print(y_pred)