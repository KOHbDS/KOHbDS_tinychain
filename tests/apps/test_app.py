import math
import numpy as np
import unittest
import testutils
import tinychain as tc

LAYER_CONFIG = [(2, 2, tc.ml.ReLU()), (2, 1, tc.ml.Sigmoid())]
LEARNING_RATE = 0.1
BATCH_SIZE = 25


class App(tc.Cluster):
    __uri__ = tc.URI("/test/app")

    def _configure(self):
        layers = [tc.ml.nn.DNNLayer.create(f"layer_{i}", *l) for i, l in enumerate(LAYER_CONFIG)]
        dnn = tc.ml.nn.Sequential.load(layers)
        self.net = tc.chain.Sync(dnn)

    @tc.get_method
    def up(self) -> tc.Bool:
        return True

    @tc.post_method
    def train(self, inputs: tc.tensor.Dense, labels: tc.tensor.Dense):
        # TODO: implement this using a background task
        pass


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_app", [App])

    def testApp(self):
        self.assertTrue(self.host.get(tc.uri(App) + "/up"))

    def testTrain(self):
        pass


def load(nparray, dtype=tc.F64):
    return tc.tensor.Dense.load(nparray.shape, dtype, nparray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
