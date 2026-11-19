import matplotlib.pyplot as plt 
import pandas  as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("csv/House Price Prediction Dataset.csv")


print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

df=pd.get_dummies(df,drop_first=True)

X=df.drop(["Price","Id"],axis=1)
y=df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

print("r2:",r2)
print("mean_square error: ",mse)
print("rmse: ",rmse)

plt.figure()
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()])
plt.show()