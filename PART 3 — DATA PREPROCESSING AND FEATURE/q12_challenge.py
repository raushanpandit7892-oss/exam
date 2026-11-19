import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("csv/titanic.csv")
print(df.head())
print(df.isnull().sum())

num_cols=df.select_dtypes(include="number").columns
cat_clos=df.select_dtypes(include="object").columns
print("number_clomuns : ",num_cols)
print("object colummns: ",cat_clos)

df_label=df.copy()
le=LabelEncoder()
for col in cat_clos:
    df_label[col]=le.fit_transform(df_label[col])
print("Encoded data",df_label.head())
df_onehot=pd.get_dummies(df,columns=cat_clos)
print("dummies columns: ", df_onehot)

x=df_onehot.drop("Survived",axis=1)
y=df_onehot["Survived"]
X_train,X_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)