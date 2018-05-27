import numpy as np
import numpy.random as rnd
import json, sys
"""
class Layer:
    # dim  -- number of neurons in the layer
    # prev -- prior layer, or None if input layer.  If None, all following
    #  params are ignored.
    # act -- activation function: accept np.array of z's, return np.array of a's
    # act_prime -- derivative function: accept np.arrays of z's and of a's,
    #  return derivative of activation wrt z's, 1-D or 2-D as appropriate
    # weights -- initial weights. Set to small random if "None".
    #
    # All the following member data are None for input layer
    # in_weights -- matrix of input weights
    # in_derivs -- derivatives of E/weight for last sample
    # zs -- z values for last sample
    # z_derivs -- derivatives of E/Z for last sample
    # batch_derivs -- cumulative sum of in_derivs across current batch
    def __init__(self, dim, prev, act, act_prime, weights=None):

    def get_dim(self):
    
    def get_deriv(self, src, trg):
   
    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals = None):

    
    # Compute self.in_derivs, assuming 
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays 
    #       of outputs and of labels, and returns an np array 
    #       of dE/da for each output
    def backpropagate(self, err_prime=None, labels=None):

    # Adjust all weights by avg gradient accumulated for current batch * -|rate|
    def apply_batch(self, batch_size, rate):
     
    # Reset internal data for start of a new batch
    def start_batch(self):

    # Add delta to the weight from src node in prior layer
    # to trg node in this layer.
    def tweak_weight(self, src, trg, delta):

    # Return string description of self for debugging
    def __repr__(self):
      

class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
    # wgts -- list of one 2-d np.array per layer in arch
    def __init__(self, arch, err, wgts = None):
    
    # Forward propagate, passing inputs to first layer, and returning outputs
    # of final layer
    def predict(self, inputs):

        
    # Assuming forward propagation is done, return current error, assuming
    # expected final layer output is |labels|
    def get_err(self, labels):

    
    # Assuming a predict was just done, update all in_derivs, and add to batch_derivs
    def backpropagate(self, labels):

    
    # Verify all partial derivatives for weights by adding an
    # epsilon value to each weight and rerunning prediction to
    # see if change in error correctly reflects weight change
    def validate_derivs(self, inputs, outputs):

    
    # Run a batch, assuming |data| holds input/output pairs comprising the batch
    # Forward propagate for each input, record error, and backpropagate.  At batch
    # end, report average error for the batch, and do a derivative update.
    def run_batch(self, data, rate):
"""


def load_config(cfg_file):
    with open(cfg_file) as config:
        config_str = config.read()
        print(config_str)


def main(cmd, cfg_file, data_file):
    commands = {"verify": 1, "run": 2}
    errors = {"cross_entropy": 1, "mse": 2}
    activations = {"relu": 1, "softmax": 2}
    # the way this is handled, the strings for the hyperparameters are
    # converted to numbers, and used in variables like command and activation
    command = commands[cmd]
    load_config(cfg_file)


main(sys.argv[1], sys.argv[2], sys.argv[3])
