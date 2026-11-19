
import pandas as pd

# 1. Load dataset
df = pd.read_csv("data.csv")

# Display original dataset
print("Original Dataset:")
print(df)

# 2. Identify missing values
print("\nMissing values:")
print(df.isnull())

# 3. Count missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# 4. Remove rows containing missing values
df_drop = df.dropna()

print("\nDataset after removing missing values:")
print(df_drop)

# 5. Impute missing values using mean
df_fill = df.fillna(df.mean(numeric_only=True))

print("\nDataset after filling missing values:")
print(df_fill)

# 6. Display cleaned dataset
print("\nCleaned Dataset:")
print(df_fill)