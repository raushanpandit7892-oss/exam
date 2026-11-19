import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("csv/Irish_Dataset_full.csv")
print(df.head())
print(df.keys())

num=df.select_dtypes(include="number")

print("Mean: ",num.mean())
print("Meadian: ",num.median())
print("Standard Deviation: ",num.std())
print("Minimum: ",num.min())
print("Maximum: ",num.max())


# for col in num.columns:
#     plt.figure()
#     sns.histplot(data=df,x=col,bins=10)
#     plt.title("Distrubation of "+col)
#     plt.show()

# for col in num.columns:
#     plt.figure()
#     sns.boxplot(data=df,y=col)
#     plt.title("box plot of"+col)
#     plt.show()


plt.figure()
sns.scatterplot(data=df,x="SepalLengthCm",y="SepalWidthCm",hue="Species")
plt.title("Sepal Lenght vs SepalWidthCm ")
plt.show()

plt.figure()
sns.heatmap(num.corr(),annot=True)
plt.title("Correction Heatmap")
plt.show()