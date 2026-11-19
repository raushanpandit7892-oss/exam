import numpy as np
arr=np.array([10,20,30,40,50,60,70,80,90,100])

total=np.sum(arr)
avrage=np.mean(arr)
maximum=np.max(arr)
minimum=np.min(arr)

addition=arr+10
subtraction=arr-20
division=arr/5
multiplication=arr*10
modulas=arr%2
power=arr**2
reminder=arr//2

print("total:",total)
print("avrage:",avrage)
print("max: ",maximum)
print("minimum: ",minimum)


print("addtion of 10 in arr: ",addition)
print("subtraction of 20 in arr: ",subtraction)
print("division of 5 in arr: ", division)
print("multiplication of 10 in arr: ",multiplication)
print("modulas of arr by 2: ",modulas)
print("power values of arr by 2: ",power)
print("reminder of arr by 2: ",reminder)
print("Dimination of array:",arr.ndim)