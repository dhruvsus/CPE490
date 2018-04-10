import numpy as np
np.set_printoptions(threshold=np.nan)
import sys
n_matrix, n_weight, values, data_input, output, input, data=[],[],[],[],[], sys.argv[1],input()
input,data='./Sample.nn','./test.dat'
for line in open(input):
    if(line.strip()):values.extend(list(map(int,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' ')))))))
for line in open(data):
    if(line.strip()):data_input.append(list(map(float,list(filter(lambda x: x!='',(line.lstrip().rstrip().split(sep=' '))))))+[1])
num_inputs,num_layers=values.pop(0),values.pop(0)
data_input,data_input_copy=np.vstack(list(map(np.asarray,data_input))),np.copy(data_input)
for i in range(num_layers):
    num_neurons=values.pop(0)
    for i in range(num_neurons):
        n_matrix,values=n_matrix+values[0:num_inputs+1],values[num_inputs+1:]
    n_weight.append(np.asarray(n_matrix).reshape(num_neurons,num_inputs+1))
    n_matrix,num_inputs=[],num_neurons
for layer in n_weight:
    output=np.heaviside(np.transpose(np.matmul(layer,np.transpose(data_input))),0.5)
    data_input=np.c_[ output, np.ones(data_input.shape[0]) ]
data_input_copy=data_input_copy[:,0:-1]
output=np.hstack((output,data_input_copy))
np.savetxt(sys.stdout, output, fmt="%.1f")
