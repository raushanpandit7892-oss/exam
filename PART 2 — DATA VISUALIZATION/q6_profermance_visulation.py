import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data={
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [30, 45, 48, 60, 62, 75, 78, 85, 88, 95]
}
df=pd.DataFrame(data)

# 1. Line plot
plt.figure()
sns.lineplot(data=df, x="Study_Hours", y="Marks", marker="o")
plt.title("Study Hours vs Marks - Line Plot")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# 2. Scatter plot
plt.figure()
sns.scatterplot(data=df, x="Study_Hours", y="Marks")
plt.title("Study Hours vs Marks - Scatter Plot")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# 3. Histogram of marks
plt.figure()
sns.histplot(data=df, x="Marks", bins=4)
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

# 4. Box plot
plt.figure()
sns.boxplot(data=df, y="Study_Hours")
plt.title("Box Plot of Study_Hours ")
plt.ylabel("Study_Hours")
plt.show()

