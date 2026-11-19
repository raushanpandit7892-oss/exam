import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_csv("csv/Advertising.csv")
print(df.head())

print("missing values: ",df.isnull().sum())

num=df.select_dtypes(include="number")
scale=StandardScaler()
standardized=scale.fit_transform(num)
standared_df=pd.DataFrame(standardized,columns=num.columns)
print("standardized data: ",standared_df.head())

ms=MinMaxScaler()
normal=ms.fit_transform(num)
normal_df=pd.DataFrame(normal,columns=num.columns)
print("Normilazied data: ",normal_df.head())