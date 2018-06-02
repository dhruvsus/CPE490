import numpy as np
import numpy.random as rnd
import json, sys


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
    def __init__(self, dim, prev, act=None, act_prime=None, weights=None):
        self.dim = dim
        self.prev = prev
        self.act = act
        self.act_prime = act_prime
        if weights is None:
            self.weights=np.random.uniform(low=-0.5,high=0.5,size=(dim,1))
        else:
            self.weights = weights
        self.outputs = None

    def get_dim(self):
        return self.dim

    #    def get_deriv(self, src, trg):
    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals = None):
        if vals is not None:
            # L0
            self.outputs=self.act(np.dot(self.weights,vals))
            return self.outputs
        else:
            # not L0
            self.outputs-self.act(np.dot(self.weights,self.prev.outputs))
            return self.outputs
    # Compute self.in_derivs, assuming
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays
    #       of outputs and of labels, and returns an np array
    #       of dE/da for each output
    """def backpropagate(self, err_prime=None, labels=None):

    # Adjust all weights by avg gradient accumulated for current batch * -|rate|
    def apply_batch(self, batch_size, rate):
     
    # Reset internal data for start of a new batch
    def start_batch(self):

    # Add delta to the weight from src node in prior layer
    # to trg node in this layer.
    def tweak_weight(self, src, trg, delta):

    # Return string description of self for debugging"""

    def __repr__(self):
        return (
            "dim: "
            + str(self.dim)
            + " act: "
            + str(self.act)
            + " act_prime: "
            + str(self.act_prime)
        )


class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
    # wgts -- list of one 2-d np.array per layer in arch
    def __init__(self, arch, err, wgts=None):
        self.layers = []
        self.arch = arch
        self.err = err
        self.wgts = wgts
        self.layers.append(Layer(dim=arch.pop(0)[0], prev=None))
        for layer_arch in arch:
            self.layers.append(
                Layer(
                    layer_arch[0],
                    prev=self.layers[0],
                    act=layer_arch[1],
                    act_prime=layer_arch[1] + "_prime",
                )
            )


# Forward propagate, passing inputs to first layer, and returning outputs
# of final layer
    def predict(self, inputs):
        # run propogate for layer 1, which doesn't have a prev
        output=None
        for layer in self.layers:
            if layer.prev is None:
                #layer is input layer ie layer 0
                output=layer.propagate(inputs)
            else:
                output=layer.propagate()
        return output
# Assuming forward propagation is done, return current error, assuming
# expected final layer output is |labels|
#    def get_err(self, labels):


# Assuming a predict was just done, update all in_derivs, and add to batch_derivs
#    def backpropagate(self, labels):


# Verify all partial derivatives for weights by adding an
# epsilon value to each weight and rerunning prediction to
# see if change in error correctly reflects weight change
#    def validate_derivs(self, inputs, outputs):


# Run a batch, assuming |data| holds input/output pairs comprising the batch
# Forward propagate for each input, record error, and backpropagate.  At batch
# end, report average error for the batch, and do a derivative update.
#    def run_batch(self, data, rate):
def relu(vals):
    """Return the numpy array from relu'ing the input vals.
    Keyword arguments:
    vals -- the input numpy array, of dimensions: batch_size, n(L-1)
    """
    return np.maximum(vals, 0)


def softmax(vals):
    """Return the numpy array from softmaxing the input values
    Keyword arguments:
    vals -- the input numpy array, of dimensions: batch_size, n(L-1)
    """
    # what it's trying to achieve: exponential for row vals/exponential sum across rows
    softmax_denominator = np.exp(vals).sum(axis=1)
    softmax_numerator = np.exp(vals)
    print(softmax_numerator)
    print(softmax_denominator)
    return softmax_numerator / softmax_denominator[:, None]


def load_config(cfg_file):
    errors = {"cross_entropy": 1, "mse": 2}
    activations = {"relu": 1, "softmax": 2}
    with open(cfg_file, "r") as config:
        config_json = json.load(config)
        num_layers = len(config_json["arch"])
        model = Network(
            arch=config_json["arch"],
            err=errors[config_json["err"]],
            wgts=[
                np.vstack(config_json.get("wgts")[i])
                for i in range(num_layers - 1)
            ],
        )
        print(model.layers)
    return model


def load_data(data_file):
    with open(data_file, "r") as data:
        data_json = json.loads(data.read())
        input = [data_json[i][0] for i in range(len(data_json))]
        input = np.vstack(input)
        output = [data_json[i][1] for i in range(len(data_json))]
        output = np.vstack(output)
        return input, output


def main(cmd, cfg_file, data_file):
    commands = {"verify": 1, "run": 2}
    # the way this is handled, the strings for the hyperparameters are
    # converted to numbers, and used in variables like command and activation
    command = commands[cmd]
    model = load_config(cfg_file)
    input, output = load_data(data_file)
    print(globals()[model.layers[1].act]([1,2,3,4,-5]))

main(sys.argv[1], sys.argv[2], sys.argv[3])
