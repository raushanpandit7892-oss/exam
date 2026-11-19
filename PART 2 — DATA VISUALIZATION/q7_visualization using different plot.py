import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df=pd.read_csv("csv/mtcars.csv")
print(df.head())

# 1. Histogram
plt.figure()
sns.histplot(data=df, x="mpg", bins=10)
plt.title("Histogram of MPG")
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.show()

# 2. Box plot
plt.figure()
sns.boxplot(data=df, y="mpg")
plt.title("Box Plot of MPG")
plt.ylabel("MPG")
plt.show()

# 3. Count plot
plt.figure()
sns.countplot(data=df, x="cyl")
plt.title("Count of Cylinders")
plt.xlabel("Cylinders")
plt.ylabel("Count")
plt.show()

# 4. Heatmap
plt.figure()
corr=df.select_dtypes(include="number").corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("COrrealtion Heatmap")
plt.show()

# 5. Pair plot
sns.pairplot(df[["mpg","hp","wt","cyl"]])
plt.show()

