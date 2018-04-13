import numpy as np
import sys
np.set_printoptions(threshold=np.nan)
nmat, nwt, values, datinput, output, input = [
], [], [], [], [], sys.argv[1]
for l in sys.stdin.readlines():
    if(l.strip()):
        datinput.append(list(map(float, list(
            filter(lambda x: x != '', (l.lstrip().rstrip()
                                       .split(sep=' ')))))) + [1])
for l in open(input):
    if(l.strip()):
        values.extend(list(map(float, list(
            filter(lambda x: x != '', (l.lstrip().rstrip().split(sep=' ')))))))
ninputs, nlayers, datinput, datinput_copy = int(values.pop(0)), int(values.pop(
    0)), np.vstack(list(map(np.asarray, datinput))), np.copy(datinput)[:, 0:-1]
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
output = np.hstack((output.astype(int), datinput_copy))
np.savetxt(sys.stdout, output, fmt="%d " + ("%0.1f ") * datinput_copy.shape[1])

