import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.linear_model import Lasso,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("csv/House Price Prediction Dataset.csv")

print(df.head(),df.info(),df.describe(),df.isnull().sum())

df=pd.get_dummies(df,drop_first=True)

X=df.drop(["Price","Id"],axis=1)
y=df["Price"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scale=StandardScaler()
X_train_scale=scale.fit_transform(X_train)
X_test_scale=scale.transform(X_test)

models={
    "LinerRegression":LinearRegression(),
    "RandomForest":RandomForestRegressor(random_state=42),
    "SVM":SVR(),
    "Lasso":Lasso()
}

result=[]

for name,model in models.items():
    model.fit(X_train_scale,y_train)
    y_pred=model.predict(X_test_scale)

    r2=r2_score(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    result.append([name,r2,rmse])

comparision=pd.DataFrame(
    result,
    columns=["MODEL","R2 Score","RMSE"]
)
print(comparision)