
import unittest
import numpy as np
from smolgrad import Tensor


class TestTensor(unittest.TestCase):

    def test_add(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        z = x + y

        self.assertTrue(np.array_equal(z.data, np.array([5, 7, 9])))
        self.assertTrue(z.requires_grad)

        z.backward()

        self.assertTrue(np.array_equal(x.grad, np.array([1, 1, 1])))
        self.assertTrue(np.array_equal(y.grad, np.array([1, 1, 1])))

    def test_mul(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        z = x * y

        self.assertTrue(np.array_equal(z.data, np.array([4, 10, 18])))
        self.assertTrue(z.requires_grad)

        z.backward()
        print(x.grad)
        self.assertTrue(np.array_equal(x.grad, np.array([4, 5, 6])))
        self.assertTrue(np.array_equal(y.grad, np.array([1, 2, 3])))


if __name__ == '__main__':
    unittest.main()
