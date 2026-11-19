
# Q4. Handling Missing Values Using Pandas

# 1. Import Pandas
import pandas as pd

# 2. Import CSV dataset
df = pd.read_csv("csv/data.csv")

# 3. Display dataset information
print("Dataset Information:")
df.info()

# 4. Identify missing values
print("\nMissing Values:")
print(df.isna().sum())

# 5. Remove rows containing missing values
df = df.dropna()

print("\nDataset after removing missing values:")
print(df)

# 6. Replace missing numerical values with mean
df = pd.read_csv("csv/data.csv")

df = df.fillna(df.mean(numeric_only=True))

# 7. Display cleaned dataset
print("\nCleaned Dataset:")
print(df)


# import pandas as pd
# df=pd.read_csv("csv/data.csv")

# print(df.head())
# print(df.info())
# print(df.keys())

# print(df.isnull().sum())
# df=df.dropna()
# print(df)

# df=pd.read_csv("csv/data_out.csv")
# df=df.fillna(df.mean(numeric_only=True))
# print(df)