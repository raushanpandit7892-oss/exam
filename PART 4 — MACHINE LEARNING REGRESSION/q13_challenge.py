import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("csv/House Price Prediction Dataset.csv")
print(df.head())
df=pd.get_dummies(df,drop_first=True)

X=df.drop("Price",axis=1)
y=df["Price"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r_square=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("prediction: ", y_pred)
print("mean square error : ",mse)
print("root mean square error : ",rmse)
print("R square; ",r_square)

plt.figure()
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()])
plt.show()
