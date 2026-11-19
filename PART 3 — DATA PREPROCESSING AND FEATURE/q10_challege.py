import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("csv/credit_dataset_full.csv")
print(df.head())
print(df.keys())

data=df["Income"]

Q1=data.quantile(0.25)
Q3=data.quantile(0.75)
IQR=Q3-Q1

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR

outliers=data[(data<lower_limit)|(data>upper_limit)]
print(outliers)
clean_data=data[~((data<lower_limit)|(data>upper_limit))]
print("clear_data:  ",clean_data)

plt.figure()
plt.boxplot(data)
plt.title("Box Plot for outliears Detection")
plt.ylabel("Values")
plt.show()

