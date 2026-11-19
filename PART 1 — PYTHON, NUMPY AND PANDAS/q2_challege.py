import pandas as pd
a=pd.Series([10,20,30,40,50])
b=pd.Series([30,40,50,60,70])

common=a[a.isin(b)]
uncommon=pd.concat([a[~a.isin(b)],
                    b[~b.isin(a)]])
print(common)
print(uncommon)
