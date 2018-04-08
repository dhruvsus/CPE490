import numpy as np
import sys
NMatrix, values, dataInput={},[],[]
print(sys.argv[1])
for line in open(sys.argv[1]):
    if(line.strip()):values.extend((list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' '))))))
#numinputs,numlayers=values.pop(0),values.pop(0)
data=input('enter file name')
for line in open(data):
    if(line.strip()):dataInput.append((list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' '))))))
print(np.asarray(dataInput))
