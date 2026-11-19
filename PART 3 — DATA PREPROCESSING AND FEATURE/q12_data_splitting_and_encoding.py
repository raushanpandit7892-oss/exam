
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("csv/data_out.csv")
print(df.head())

# 2. Identify numerical and categorical columns
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# 3. Label Encoding
df_label = df.copy()

le = LabelEncoder()

for col in cat_cols:
    df_label[col] = le.fit_transform(df_label[col])

print("\nLabel Encoded Dataset:")
print(df_label.head())

# 4. One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=cat_cols)

print("\nOne-Hot Encoded Dataset:")
print(df_onehot.head())

# 5. Split dataset
X = df_onehot
y = df_onehot.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Display shapes
print("\nTraining X shape:", X_train.shape)
print("Testing X shape:", X_test.shape)
print("Training y shape:", y_train.shape)
print("Testing y shape:", y_test.shape)
