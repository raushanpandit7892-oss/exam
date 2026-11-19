import pandas as pd
a=pd.Series([1,2,3,4,5])
b=pd.Series([4,5,6,7,8])
comman=a[a.isin(b)]
uncommon=pd.concat([a[~a.isin(b)],
                    b[~b.isin(a)]])

print("A: ",a)
print("\n B:",b)
print("\n common elements:",comman.tolist())
print("\n uncommon elements: ",uncommon.tolist())
