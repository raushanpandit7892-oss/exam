import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("csv/mtcars.csv")
print(df.head())

corr=df.corr(numeric_only=True,)
sns.heatmap(corr,annot=True)
plt.show()

sns.pairplot(df[["mpg","hp","cyl","wt"]])
plt.show()


plt.figure()
sns.histplot(data=df, x="hp", bins=10)
plt.title("Histogram of MPG")
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.show()


plt.figure()
sns.boxplot(data=df, y="wt")
plt.title("Box Plot of MPG")
plt.ylabel("MPG")
plt.show()


plt.figure()
sns.countplot(data=df, x="am")
plt.title("Count of Cylinders")
plt.xlabel("Cylinders")
plt.ylabel("Count")
plt.show()
