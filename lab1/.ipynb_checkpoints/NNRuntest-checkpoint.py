import numpy as np
import sys
np.set_printoptions(threshold=np.nan)
nmat, nwt, values, datinput, output, input, data = [
], [], [], [], [], 'Answer.nn', 'untrain.dat'
for l in open(input):
    if(l.strip()):
        values.extend(list(map(float, list(
            filter(lambda x: x != '', (l.lstrip().rstrip().split(sep=' ')))))))
for l in open(data):
    if(l.strip()):
        datinput.append(list(map(float, list(
            filter(lambda x: x != '', (l.lstrip().rstrip().split(sep=' ')))))) + [1])
ninputs, nlayers, datinput, datinput_copy= int(values.pop(0)), int(
    values.pop(0)), np.vstack(list(map(np.asarray, datinput))), np.copy(datinput)[:, 0:-1]
for i in range(int(nlayers)):
    nneurons = int(values.pop(0))
    for i in range(nneurons):
        nmat, values = nmat + values[0:ninputs + 1], values[ninputs + 1:]
    nwt.append(np.asarray(nmat).reshape(nneurons, ninputs + 1))
    nmat, ninputs = [], nneurons
for layer in nwt:
    output = np.heaviside(np.transpose(
        np.matmul(layer, np.transpose(datinput))), 0)
    datinput = np.c_[output, np.ones(datinput.shape[0])]
for answer, original_input in zip(output.astype(int), datinput_copy):
    np.savetxt(sys.stdout, answer, fmt="%d", newline=" ")
    np.savetxt(sys.stdout, original_input, fmt="%.1f",newline=" ")
    print()
