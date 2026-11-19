import pandas as pd
df=pd.read_csv("csv/student.csv")
print("Information of the dataset: ",df.info())
print("Missing values in the dataset: ",df.isnull().sum())
df_mean=df.fillna(df.mean(numeric_only=True))
print("replacing missing values with mean: ",df_mean)
