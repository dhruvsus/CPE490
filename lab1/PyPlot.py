import numpy as np
import matplotlib.pyplot as plt
data,data_input='train.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
data_input=np.array(data_input)
x=data_input[:,1:2]
y=data_input[:,2:3]
plt.scatter(y,x,c=data_input[:0:1])
plt.show()
