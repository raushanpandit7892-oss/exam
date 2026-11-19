
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 1 & 2. Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90, 95, 98]
}

df = pd.DataFrame(data)
print(df)

# 3. Separate independent and dependent variables
X = df[["Study_Hours"]]
y = df["Marks"]

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Build Linear Regression model
model = LinearRegression()

# 6. Train model
model.fit(X_train, y_train)

# 7. Predict marks
y_pred = model.predict(X_test)
print("Predicted Marks:", y_pred)

# 8. Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)

# 9. Visualize actual and predicted values
plt.figure()
plt.scatter(X_test, y_test, label="Actual Marks")
plt.scatter(X_test, y_pred, label="Predicted Marks")
plt.plot(X_test, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Actual vs Predicted Marks")
plt.legend()
plt.show()
