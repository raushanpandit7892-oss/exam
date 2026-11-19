import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("csv/Irish_Dataset_full.csv")
print(df.head())

num=df.select_dtypes(include="number")

for col in num.columns:
    plt.figure()
    sns.boxplot(data=df,y=col)
    plt.show()
for col in num.columns:
    plt.figure()
    sns.histplot(data=df,x=col,bins=10)
    plt.show()

sns.scatterplot(data=df,x="PetalWidthCm",y="SepalLengthCm",hue="Species")
plt.show()

sns.heatmap(num.corr(),annot=True)
plt.show()

print("Mean: ",num.mean())
print("Meadian: ",num.median())
print("Standard Deviation: ",num.std() )
print("minimum: ",num.min())
print("Maximum: ",num.max())