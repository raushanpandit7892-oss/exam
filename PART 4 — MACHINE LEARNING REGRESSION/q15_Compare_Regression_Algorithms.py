import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error

df=pd.read_csv("csv/House Price Prediction Dataset.csv")
print(df.head())
print(df.info(),df.describe(),df.isnull().sum())

df=pd.get_dummies(df,drop_first=True)

X=df.drop(["Price","Id"],axis=1)
y=df["Price"]

X_train ,X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
Scale=StandardScaler()
X_train_scale=Scale.fit_transform(X_train)
X_test_scale=Scale.transform(X_test)


models={
    "Linear Regression":LinearRegression(),
    "Random Forest":RandomForestRegressor(random_state=42),
    "SVR":SVR(),
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
    columns=["Model","R2 Score","RMSE"]
)
print(comparision)