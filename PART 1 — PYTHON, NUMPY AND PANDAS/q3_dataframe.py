import pandas as pd
data={
    "Employee_names":["rahul","priya","amit","sneha","karan","neha","arjun"],
    "Department":["IT","HR","IT","Finance","HR","Finance","IT"],
    "Employee_Type":["permanent","permanent","contract","permanent","contract","contract","permanent"],
    "Employee_salary":[50000,40000,30000,45000,25000,35000,60000]
}
df=pd.DataFrame(data)
print(df.head())
print("statistical information.: ",df.describe())
print("Group employees by department.: ",df.groupby("Department").size())
print(" Calculate average salary.: ",df.groupby("Department")["Employee_salary"].mean())

pivot=pd.pivot_table(df,
                     values="Employee_salary",
                     index="Department",
                     columns="Employee_Type",
                     aggfunc="mean"
                     )
print(pivot)