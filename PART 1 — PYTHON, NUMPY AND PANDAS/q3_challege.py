import pandas as pd
data={
    "Employee_names":["rahul","priya","amit","sneha","karan","neha","arjun"],
    "Department":["IT","HR","IT","Finance","HR","Finance","IT"],
    "Employee_Type":["permanent","permanent","contract","permanent","contract","contract","permanent"],
    "Employee_salary":[50000,40000,30000,45000,25000,35000,60000]
}
df=pd.DataFrame(data)
print(df.head(3))
print(df.describe())
print(df.groupby("Employee_Type").size())
print(df.groupby("Employee_Type")["Employee_salary"].sum())

pivot=pd.pivot_table(
    df,
    index="Employee_Type",
    columns="Department",
    values="Employee_salary",
    aggfunc="mean"
)
print(pivot)