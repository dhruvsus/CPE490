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
    # outputs -- output for layer.
    # All the following member data are None for input layer
    # in_weights -- matrix of input weights
    # in_derivs -- derivatives of E/weight for last sample
    # zs -- z values for last sample
    # z_derivs -- derivatives of E/Z for last sample
    # batch_derivs -- cumulative sum of in_derivs across current batch
    def __init__(self, dim, prev, act, act_prime, weights=None):
        self.dim = dim
        self.prev = prev
        self.act = act
        self.act_prime = act_prime
        if self.prev is None:
            # input layer
            self.weights = weights
        else:
            self.weights = (np.random.uniform(
                low=-0.5, high=0.5, size=(dim, self.prev.dim + 1))
                            if weights is None else weights)
        self.in_weights = self.weights
        # this includes the weights field in the input layer
        self.outputs = None
        self.zs = None

    def get_dim(self):
        return self.dim

    def get_deriv(self, src, trg):
        pass

    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals=None):
        if vals is not None:
            # input layer
            self.outputs = vals
        else:
            inputs = np.reshape(self.prev.outputs, newshape=(-1, 1))
            inputs = np.append(inputs, 1)
            weights = np.vstack(self.in_weights)
            outputs = np.dot(self.in_weights, inputs)
            self.zs = outputs
            outputs = globals()[self.act](outputs)
            # print(outputs)
            self.outputs = outputs

    # Compute self.in_derivs, assuming
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays
    #       of outputs and of labels, and returns an np array
    #       of dE/da for each output
    def backpropagate(self, err_prime=None, labels=None):
        pass

    # Adjust all weights by avg gradient accumulated for current batch * -|rate|
    def apply_batch(self, batch_size, rate):
        pass

    # Reset internal data for start of a new batch
    def start_batch(self):
        pass

    # Add delta to the weight from src node in prior layer
    # to trg node in this layer.
    def tweak_weight(self, src, trg, delta):
        pass

    # Return string description of self for debugging
    def __repr__(self):
        return "dim: {!s}\nact: {}\nact_prime: {}\nweights: {!s}\n".format(
            self.dim, self.act, self.act_prime, self.weights)


def relu(inputs):
    return np.maximum(inputs, 0)


def softmax(inputs):
    return np.exp(inputs) / np.sum(np.exp(inputs))


def cross_entropy(x, y):
    y_0 = np.nonzero(y)
    return -1 * (
        np.sum(y[y_0] * np.log(x[y_0])) - np.sum(y[y_0] * np.log(y[y_0])))


def mse(x, y):
    return np.sum(np.square(y - x)) / len(x)


class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
    # wgts -- list of one 2-d np.array per layer in arch
    # layers -- list of Layer objects
    def __init__(self, arch, err, wgts=None):
        layers = []
        self.arch = arch
        self.err = err
        # handling no weights being provided
        if wgts == None:
            self.wgts = [None]
        else:
            self.wgts = [None] + wgts
        # now to create the random weights if they don't exist.
        for layer_no, layer_arch in enumerate(arch):
            # layer no 0: input
            layers.append(
                Layer(
                    dim=layer_arch[0],
                    prev=None if layer_no == 0 else layers[-1],
                    act=layer_arch[1],
                    act_prime=layer_arch[1] + "_prime",
                    weights=None
                    if len(self.wgts) < layer_no + 1 else self.wgts[layer_no],
                ))
        self.layers = layers

    # Forward propagate, passing inputs to first layer, and returning outputs
    # of final layer
    def predict(self, inputs):
        # for input layer
        for layer_no, layer in enumerate(self.layers):
            layer.propagate(
                vals=inputs) if layer_no == 0 else layer.propagate()
        return self.layers[-1].outputs

    # Assuming forward propagation is done, return current error, assuming
    # expected final layer output is |labels|
    def get_err(self, labels):
        # print("x = {}".format(self.layers[-1].outputs))
        # print("y = {}".format(labels))
        return globals()[self.err](self.layers[-1].outputs, labels)

    # Assuming a predict was just done, update all in_derivs, and add to batch_derivs
    def backpropagate(self, labels):
        pass

    # Verify all partial derivatives for weights by adding an
    # epsilon value to each weight and rerunning prediction to
    # see if change in error correctly reflects weight change
    def validate_derivs(self, inputs, outputs):
        pass

    # Run a batch, assuming |data| holds input/output pairs comprising the batch
    # Forward propagate for each input, record error, and backpropagate.  At batch
    # end, report average error for the batch, and do a derivative update.
    def run_batch(self, data, rate):
        inputs = data["inputs"]
        outputs = data["outputs"]
        err = 0
        for input_no, input in enumerate(inputs):
            output_obs = self.predict(input)
            truth_value = outputs[input_no]
            sample_err = self.get_err(truth_value)
            # print(truth_value)
            print("{} vs {} for {:0.6f}".format(output_obs, truth_value,
                                                sample_err))
            err = err + self.get_err(truth_value)
        err = err / len(inputs)
        print(err)


def load_config(cfg_file):
    with open(cfg_file, "r") as config:
        config_json = json.load(config)
        arch = config_json["arch"]
        err = config_json["err"]
        wgts = config_json.get("wgts")
        model = Network(arch=arch, err=err, wgts=wgts)
        # print(model.layers)
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
    #testing area
    # print(relu(np.asarray([5,6,-1,2,4,-7])))
    # relu works
    #end testing area
    model = load_config(cfg_file)
    inputs, outputs = load_data(data_file)
    data = {"inputs": inputs, "outputs": outputs}
    model.run_batch(data=data, rate=0.01)


main(sys.argv[1], sys.argv[2], sys.argv[3])
