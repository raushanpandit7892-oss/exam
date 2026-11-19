import  pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df=pd.read_csv("csv/mtcars.csv")
print(df.keys())

plt.figure()
sns.histplot(data=df,x=df["hp"],bins=8)
plt.title("Frequency of Hp")
plt.xlabel("hp")
plt.ylabel("frequency")
plt.show()

plt.figure()
sns.scatterplot(data=df,x="hp",y="mpg")
plt.title("hp vs mpg")
plt.xlabel("hp")
plt.ylabel("mpg")
plt.show()

plt.figure()
sns.barplot(data=df,x="cyl")
plt.show()

plt.figure()
plt.boxplot(data=df,x="wt")
plt.show()