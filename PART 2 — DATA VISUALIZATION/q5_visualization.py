import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("csv/mtcars.csv")
print(df.head())

# 1. Histogram for mpg
plt.figure()
sns.histplot(data=df,x="mpg",bins=10)
plt.xlabel("Miles per Gallon")
plt.ylabel("Frequency")
plt.title("Histogram of MPG")
plt.show()


# 2. Scatter plot between wt and mpg
plt.figure()
sns.scatterplot(data=df,x="wt",y="mpg")
plt.title("Weight vs MPG")
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.show()


# 4. Box plot for mpg
plt.figure()
sns.barplot(data= df,x="mpg")
plt.title("Box plot of MPG")
plt.ylabel("Mile per Gallon")
plt.show()

# 3. Bar plot for transmission type
plt.figure()
sns.countplot(data=df,x="am")
plt.title("transmission type")
plt.xlabel("Transmission (0=Automatic, 1=Manual)")
plt.ylabel("count")
plt.show()