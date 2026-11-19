import pandas as pd
df=pd.read_csv("csv/data_out.csv")
print(df.head())
print("missing values: ",df.isnull().sum())
print("drope null values: ",df.dropna())
print("replace missing values with mean: ",df.fillna(df.mean(numeric_only=True)))
