
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1 & 2. Load dataset
df = pd.read_csv("csv/data_out.csv")
print("Original Dataset:")
print(df.head())

# Select numerical columns
num = df.select_dtypes(include="number")

# 3. Apply standardization
ss = StandardScaler()
standardized = ss.fit_transform(num)

standardized_df = pd.DataFrame(standardized, columns=num.columns)

print("\nStandardized Data:")
print(standardized_df.head())

# 4. Apply normalization
ms = MinMaxScaler()
normalized = ms.fit_transform(num)

normalized_df = pd.DataFrame(normalized, columns=num.columns)

print("\nNormalized Data:")
print(normalized_df.head())

# 5. Display transformed values
print("\nStandardized Values:")
print(standardized_df)

print("\nNormalized Values:")
print(normalized_df)

# 6. Compare results
print("\nComparison:")
print("Original Data:")
print(num.head())

print("\nStandardization:")
print(standardized_df.head())

print("\nNormalization:")
print(normalized_df.head())
