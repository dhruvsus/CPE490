#Lab 5 Backpropagation#

## Edits

## Overview
In this lab you'll implement a basic backpropagation algorithm.

## Model Design
A model design for a Layer and Network class comes with the lab.  This is the skeleton of the working reference application.  You may adjust it modestly if you like, but any serious deviation must first be OKed by me.  This is a difficult algorithm to design well, and this starting point will help.

## Application Overview
Write a single Python application BackProp.py, that accepts three commandline arguments: a command "run" or "verify", and two files, the first a network configuration file and the second a file of input/output pairs.

### Configuration file format
Two example configuration files are given.  Each contains a JSON object with these properties:

* *arch* an array of pairs describing the layers of a network, including its dimension and activation.  The input layer has no activation function.  Other layers may have "relu" or "softmax".  
* *err* either "cross_entropy" or "mse", describing the error function.
* *weights* (optional) an array of 2-d weight arrays, one for each layer other than the input layer.  Note that each has a different dimension, appropriate to its layer.  This means you can't convert the whole thing into one big numpy array, though each element may be so converted.

If *weights* is present, use these initial values for the network weights.  Otherwise assign small values between -.5 and .5 to all weights.

The file CfgEx contains a configuration matching our in-class example, including initial weights.

### Data file format
Two example data files are given.  Each contains a JSON object describing an array of input/output pairs.  Each input and output is an array of values of dimension appropriate for the corresponding Cfg file.  The DataEx file contains a single input/output pair matching the in-class example.

## The Commands
The two commands *run* and *verify* operate thus:

### Verify
The verify command is the first you should implement.  For each input/output pair (possibly just one) it does the following:

 1. Runs a forward propagation to establish an output value and corresponding error value.  
 2. Reports the actual output vs the expected output, and the resultant error.
 3. Runs a single backpropagation to determine dE/dW values.
 3. Systematically, for each weight at each level, adjusts that weight by .01, reruns the forward propagation to get a new error value, and reports the difference and the ratio of the difference to the expected difference (.01 * dE/dW for the adjusted weight) as a percentage, to 4 decimals. 
 4. Returns the weight to its original value. 

Here's a sample verify run:
<pre>
$ python BackProp.py verify CfgEx DataEx
[0.61097505 0.01844985 0.3705751 ]  vs  [0.3 0.  0.7]  for  0.2318348512774041
Test 0/0 to 1/0: 0.244284 - 0.231835 = 0.012449 (0.0102% error)
Test 0/1 to 1/0: 0.257121 - 0.231835 = 0.025286 (0.0103% error)
Test 0/2 to 1/0: 0.244284 - 0.231835 = 0.012449 (0.0102% error)
Test 0/0 to 1/1: 0.231835 - 0.231835 = 0.000000 (0.0000% error)
Test 0/1 to 1/1: 0.231835 - 0.231835 = 0.000000 (0.0000% error)
Test 0/2 to 1/1: 0.231835 - 0.231835 = 0.000000 (0.0000% error)
Test 0/0 to 1/2: 0.236802 - 0.231835 = 0.004968 (0.0101% error)
Test 0/1 to 1/2: 0.241822 - 0.231835 = 0.009987 (0.0101% error)
Test 0/2 to 1/2: 0.236802 - 0.231835 = 0.004968 (0.0101% error)
Test 1/0 to 2/0: 0.233393 - 0.231835 = 0.001558 (0.0100% error)
Test 1/1 to 2/0: 0.231835 - 0.231835 = 0.000000 (0.0000% error)
Test 1/2 to 2/0: 0.234956 - 0.231835 = 0.003122 (0.0100% error)
Test 1/3 to 2/0: 0.234956 - 0.231835 = 0.003122 (0.0100% error)
Test 1/0 to 2/1: 0.231927 - 0.231835 = 0.000092 (0.0100% error)
Test 1/1 to 2/1: 0.231835 - 0.231835 = 0.000000 (0.0000% error)
Test 1/2 to 2/1: 0.232020 - 0.231835 = 0.000185 (0.0100% error)
Test 1/3 to 2/1: 0.232020 - 0.231835 = 0.000185 (0.0100% error)
Test 1/0 to 2/2: 0.230191 - 0.231835 = -0.001644 (-0.0100% error)
Test 1/1 to 2/2: 0.231835 - 0.231835 = 0.000000 (0.0000% error)
Test 1/2 to 2/2: 0.228552 - 0.231835 = -0.003283 (-0.0100% error)
Test 1/3 to 2/2: 0.228552 - 0.231835 = -0.003283 (-0.0100% error)
</pre>

### Run
The run command:
 1. Sets up the network per config file
 2. Trains it on the first 3/4 of the samples in the data file, in 32-sample batches, for one epoch.
 3. Reports the range of samples per batch and the average error across the samples in the batch
 4. Runs each of the remaining 1/4 of the samples as validation, and reports the average error across all the validation samples.

Here's a sample run:

<pre>
$ python BackProp.py run CfgTrain DataTrain
Batch 0:32
Batch error: 12.116
Batch 32:64
Batch error: 7.197
Batch 64:96
Batch error: 3.051
Batch 96:128
Batch error: 1.777
Batch 128:150
Batch error: 0.701
Validation error: 0.786
</pre>

## Error Checking
Assume the files are properly formatted, the commandline is properly formed, and the config file and data file match one another.  This is about getting the computation right, not detailed input data checking.

## Code Elegance and Numpy
We won't enforce a strict elegance requirement, but do your best to use Numpy's considerable features for this assignment.  Some concrete stats on the reference implementation:

1. 12 loops total, including all I/O
2. 2 uses of np.outer
3. 3 uses of np.dot
3. 1 use of np.diag
4. 1 use of np.transpose
5. 3 uses of np.sum
6. 2 uses of np.log
7. 1 use of np.exp
8. 227 lines total, including about 75 comment/blank lines

## Turnin
Submit the following files:

<pre>
BackProp.py
</pre>

to turnin directory ~grade-cstaley/DLNN/BackProp/turnin on the 127 machines.  
