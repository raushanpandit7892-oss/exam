
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load numerical dataset
df = pd.read_csv("csv/data_out.csv")

# Select numerical column
data = df["column_name"]

# 2. Calculate Q1 and Q3
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

# 3. Calculate IQR
IQR = Q3 - Q1

# 4. Calculate lower and upper limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# 5. Identify outliers
outliers = data[(data < lower_limit) | (data > upper_limit)]

print("Outliers:")
print(outliers)

# 6. Visualize using box plot
plt.figure()
plt.boxplot(data)
plt.title("Box Plot for Outlier Detection")
plt.ylabel("Values")
plt.show()
