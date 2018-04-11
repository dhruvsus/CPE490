import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
data,data_input='train.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
data_input=np.array(data_input)
x=data_input[:,1:2]
y=data_input[:,2:3]
plt.scatter(x,y,c=data_input[:,0:1])
plt.axhline(0, color='blue')
plt.axvline(0, color='blue')
plt.show()
data,data_input='Answer.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
data_input=np.array(data_input)
x=data_input[:,1:2]
y=data_input[:,2:3]
plt.scatter(x,y,c=data_input[:,0:1])
plt.axhline(0, color='blue')
plt.axvline(0, color='blue')
plt.show()
data,data_input='Answer.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
data_input=np.array(data_input)
a=np.sort(data_input,axis=0)
data,data_input='train.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
data_input=np.array(data_input)
b=np.sort(data_input,axis=0)
for dataA,dataB in zip(a,b):
    if(np.not_equal(dataA,dataB).all()):
        plt.plot(dataA[1], dataA[2], marker='o', markersize=5, color="red")
        plt.plot(dataB[1], dataB[2], marker='o', markersize=5, color="green")
plt.title("Points of difference")
plt.xlabel("x")
plt.ylabel("y")
plt.show()