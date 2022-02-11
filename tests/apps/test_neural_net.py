import numpy as np
import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/neural_net")


# TODO: support abstract methods
class Differentiable(tc.Interface):
    @tc.post_method
    def forward(self, inputs) -> tc.tensor.Tensor:
        pass

    # TODO: support type hints using typing.Tuple, e.g. Tuple[Tensor, Tuple[Parameter]]
    @tc.post_method
    def backward(self, inputs) -> tc.Tuple:
        pass


# TODO: add support for @classmethod and @staticmethod to app.model
class Activation(tc.app.Model, Differentiable):
    __uri__ = URI + "/Activation"

    @tc.get_method
    def optimal_std(self, shape: tc.Tuple):
        """Calculate the recommended standard deviation for a trainable parameter using this `Activation`."""

        input_size, output_size = [tc.UInt(dim) for dim in shape.unpack(2)]
        return (input_size * output_size)**0.5

    @tc.post_method
    def forward(self, _inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        return tc.error.NotImplemented("Activation is an abstract class")

    @tc.post_method
    def backward(self, _inputs: tc.tensor.Tensor) -> tc.Tuple:
        return tc.error.NotImplemented("Activation is an abstract class")


class Sigmoid(Activation):
    """Sigmoid activation function"""

    __uri__ = URI + "/Sigmoid"

    @tc.get_method
    def optimal_std(self, shape: tc.Tuple):
        """Calculate the standard deviation for Xavier initialization for use with this :class:`Activation` function."""

        input_size, output_size = [tc.UInt(dim) for dim in shape.unpack(2)]
        return 1.0 * (2 / (input_size + output_size))**0.5

    @tc.post_method
    def forward(self, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        return 1 / (1 + (-inputs).exp())

    @tc.post_method
    def backward(self, inputs: tc.tensor.Tensor) -> tc.Tuple:
        sig = self.forward(inputs=inputs)  # TODO: cache this value
        return sig * (1 - sig)


class Trainer(tc.app.Library):
    __uri__ = URI

    @staticmethod
    def exports():
        return [
            Sigmoid,
        ]

    @tc.post_method
    def forward(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.activation = self.Sigmoid(None)
        return cxt.activation.forward(inputs=inputs)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_neural_net", [Trainer()])

    def testForward(self):
        print(self.host.post(tc.uri(Trainer) + "/forward", {"inputs": load(np.ones([1]))}))


def load(nparray, dtype=tc.F32):
    return tc.tensor.Dense.load(nparray.shape, dtype, nparray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
