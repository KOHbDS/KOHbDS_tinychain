import unittest
import testutils
import tinychain as tc


LAYER_CONFIG = [(2, 2, tc.ml.ReLU()), (2, 1, tc.ml.Sigmoid())]
LEARNING_RATE = 0.1
BATCH_SIZE = 25


URI = tc.URI("/test/neural_net")


class Layer(tc.Map, metaclass=tc.Meta):
    ERR_BASE_CLASS = tc.String("this is a base class--consider DNNLayer or ConvLayer instead")
    __uri__ = URI.append("Layer")

    @tc.put_method
    def reset(self):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)

    @tc.post_method
    def forward(self, inputs):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)

    @tc.post_method
    def backward(self, inputs, loss):
        return tc.error.NotImplemented(Layer.ERR_BASE_CLASS)


class DNNLayer(Layer):
    __uri__ = URI.append("DNNLayer")


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

    @tc.post_method
    def forward(self, inputs):
        return inputs


class Trainer(tc.Cluster):
    __uri__ = URI

    def _configure(self):
        self.Layer = Layer
        self.DNNLayer = DNNLayer
        self.NeuralNet = NeuralNet
        self.Sequential = Sequential


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_neural_net", [Trainer])

    def testUp(self):
        print(self.host.get(tc.uri(Trainer) + "/Sequential"))


if __name__ == "__main__":
    unittest.main()
