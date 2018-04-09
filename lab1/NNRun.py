import numpy as np
import sys
NMatrix, Nweight, values, dataInput,values,data=[],[],[],[],[],input('enter file name')
for line in open(sys.argv[1]):
    if(line.strip()):values.extend(list(map(int,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
for line in open(data):
    if(line.strip()):dataInput.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
numinputs,numlayers=values.pop(0),values.pop(0)
for i in range(numlayers):
    numNeurons=values.pop(0)
    for i in range(numNeurons):
        NMatrix,values=NMatrix+values[0:numinputs+1],values[numinputs+1:]
    Nweight.append(np.asarray(NMatrix).reshape(numNeurons,numinputs+1))
    NMatrix,numinputs=[],numNeurons
print(Nweight)
print(dataInput)
