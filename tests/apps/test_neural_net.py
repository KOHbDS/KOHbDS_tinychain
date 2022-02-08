import numpy as np
import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/neural_net")


class Sigmoid(object):
    """Sigmoid activation function"""

    __uri__ = URI + "/sigmoid"

    @staticmethod
    def std_initializer(input_size, output_size):
        """Calculate the standard deviation for Xavier initialization for use with this :class:`Activation` function."""

        return 1.0 * (2 / (input_size + output_size))**0.5

    @staticmethod
    def forward(inputs: tc.tensor.Tensor):
        return 1 / (1 + (-inputs).exp())

    @staticmethod
    def backward(inputs: tc.tensor.Tensor):
        sig = 1 / (1 + (-inputs).exp())  # TODO: cache this value
        return sig * (1 - sig)


# TODO: refactor into a model when ORM is implemented
class Layer(tc.Map, metaclass=tc.Meta):
    ERR_BASE_CLASS = tc.String("this is a base class--consider DNNLayer or ConvLayer instead")
    __uri__ = URI.append("Layer")

    @tc.put_method
    def reset(self):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)

    @tc.post_method
    def forward(self, inputs: tc.tensor.Tensor):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)

    @staticmethod
    @tc.post_method
    def backward(self, inputs, loss):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)


class DNNLayer(Layer):
    __uri__ = URI.append("DNNLayer")

    @classmethod
    def create(cls, input_size, output_size):
        """
        Create a new, empty `DNNLayer` with the given shape and activation function.

        Args:
            `input_size`: size of inputs;
            `output_size`: size of outputs;
            `activation`: activation function.
        """

        weights = tc.tensor.Dense.create([input_size, output_size])
        bias = tc.tensor.Dense.create([output_size])
        return cls.load(weights, bias)

    @classmethod
    def load(cls, weights, bias):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""

        return cls({"weights": weights, "bias": bias})

    @tc.post_method
    def forward(self, inputs):
        inputs = tc.tensor.einsum("ki,ij->kj", [inputs, self["weights"]]) + self["bias"]
        return Sigmoid.forward(inputs=inputs)


# TODO: refactor into a model when ORM is implemented
class NeuralNet(tc.Tuple, metaclass=tc.Meta):
    __uri__ = URI.append("NeuralNet")

    @tc.put_method
    def reset(self):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)

    @tc.post_method
    def forward(self, inputs):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)

    @tc.post_method
    def backward(self, inputs, loss):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)


class Sequential(NeuralNet):
    __uri__ = URI.append("Sequential")

    @classmethod
    def load(cls, layers):
        if not layers:
            raise ValueError("cannot initialize a neural net with no layers")

        return cls(layers)

    @tc.post_method
    def forward(self, inputs):
        return inputs


LAYER_CONFIG = [(2, 2), (2, 1)]
LEARNING_RATE = 0.1
BATCH_SIZE = 25


class Trainer(tc.app.App):
    __uri__ = URI

    @staticmethod
    def exports():
        return [
            Layer,
            DNNLayer,
            NeuralNet,
            Sequential,
        ]

    def __init__(self):
        layers = [DNNLayer.create(*l) for l in LAYER_CONFIG]
        dnn = Sequential.load(layers)
        self.net = tc.chain.Sync(dnn)
        tc.app.App.__init__(self)

    # @tc.post_method
    # def eval(self, inputs: tc.tensor.Tensor):
    #     return self.net.forward(inputs=inputs)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_neural_net", [Trainer()])

    def testUp(self):
        print(self.host.get(tc.uri(Trainer) + "/net"))

    @unittest.skip
    def testEval(self):
        inputs = np.random.random(BATCH_SIZE * 2).reshape([BATCH_SIZE, 2])
        print(self.host.post(tc.uri(Trainer) + "/eval", {"inputs": load(inputs)}))


def load(nparray, dtype=tc.F32):
    return tc.tensor.Dense.load(nparray.shape, dtype, nparray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
