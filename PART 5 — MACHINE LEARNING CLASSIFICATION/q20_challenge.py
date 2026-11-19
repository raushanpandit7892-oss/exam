import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import(
    recall_score,
    accuracy_score,
    precision_score,
    f1_score
)

df=load_breast_cancer()
X=pd.DataFrame(df.data,columns=df.feature_names)
y=df.target

print(X.head())

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

models={
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "SVM":SVC(),
    "LogisticRegression":LogisticRegression(),
    "randomforest":RandomForestClassifier(n_estimators=100,random_state=42)

}
result=[]
for name,model in models.items():
    model.fit(X_train_scale,y_train)
    y_pred=model.predict(X_test_scale)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    result.append([name,recall,precision,f1,accuracy])

comarision=pd.DataFrame(
    result,
    columns=["Model","Accuracy","precision","recall","f1_score"]
)
print(comarision)