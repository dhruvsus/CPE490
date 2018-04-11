import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
def draw_plot(xmin,xmax,ymin,ymax,x,y,color,ax=0,ay=0,xlabel="",ylabel="",title=""):
    plt.scatter(x,y,c=color)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(ax, color='blue')
    plt.axvline(ay, color='blue')
    plt.show()
data,data_input='train.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
a=np.array(data_input)
#a=np.sort(data_input,axis=0)
data,data_input='Answer.dat',[]
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
b=np.array(data_input)
#b=np.sort(data_input,axis=0)
#a is the train.dat, b is the Answer.dat
draw_plot(-100,100,-100,100,a[:,1:2],a[:,2:3],a[:,0:1])
draw_plot(-100,100,-100,100,b[:,1:2],b[:,2:3],b[:,0:1])
accuracy=np.sum(np.absolute(a[:,0:1]-b[:,0:1]))/a.shape[0]
print(100-accuracy*100)
c=[]
for dataA,dataB in zip(a,b):
    if(np.not_equal(dataA,dataB).any()):
        c.append([dataA,dataB])
        plt.plot(dataA[1], dataA[2], marker='o', markersize=5, color="red")
        #plt.plot(dataB[1], dataB[2], marker='o', markersize=5, color="green")
print(len(c))
plt.title("Points of difference")
plt.axhline(0, color='blue')
plt.axvline(0, color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
draw_plot(0,100,0,100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="First Quadrant Train")
draw_plot(0,100,0,100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="First Quadrant Answer")
draw_plot(0,-100,0,100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="Second Quadrant Train")
draw_plot(0,-100,0,100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="Second Quadrant Answer")
draw_plot(0,-100,0,-100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="Third Quadrant Train")
draw_plot(0,-100,0,-100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="Third Quadrant Answer")
draw_plot(0,100,0,-100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="Fourth Quadrant Train")
draw_plot(0,100,0,-100,a[:,1:2],a[:,2:3],a[:,0:1],xlabel="X",ylabel="Y",title="Fourth Quadrant Answer")