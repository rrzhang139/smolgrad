import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        topo = []
        visited = set()

        def sort(node):
            if node not in visited:
                visited.add(node)
                for operand in node._prev:
                    sort(operand)
                topo.append(node)
        sort(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            # Backpropogating and updating gradients using chain rule
            v._backward()

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data,
                        requires_grad=self.requires_grad or other.requires_grad)

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += result.grad  # d(result)/d(self)
                if other.requires_grad:
                    other.grad += result.grad

            result._backward = _backward
            result._prev = {self, other}

        return result

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data,
                        requires_grad=self.requires_grad or other.requires_grad)

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += result.grad * other.data
                if other.requires_grad:
                    other.grad += result.grad * self.data

            result._backward = _backward
            result._prev = {self, other}

        return result

    def __matmul__(self, other) -> "Tensor":
        result = Tensor(self.data @ other.data,
                        requires_grad=self.requires_grad or other.requires_grad)

        if result.requires_grad:
            le_axis = (0,) if self.data.ndim == 1 else ()
            re_axis = (-1,) if other.data.ndim == 1 else ()

            result_axis = le_axis + re_axis

            # When performing matrix multiplication, there could be dimensions before it.
            # Important to try leave the dimensions of the matrix mul for now, and just broadcast the outer dims
            # inputting dims in tuple
            l_indices, r_indices = broadcast(
                self.data.shape[:-2], other.data.shape[:-2])

            def _backward():
                if self.requires_grad:
                    self.grad = np.reshape(
                        np.sum(
                            np.expand_dims(result.grad, axis=result_axis) @
                            np.expand_dims(
                                other.data, axis=re_axis).swapaxes(-1, -2),
                            axis=l_indices
                        ),
                        newshape=self.grad.shape
                    )
                if other.requires_grad:
                    other.grad = np.reshape(
                        np.sum(
                            np.expand_dims(
                                self.data, axis=le_axis).swapaxes(-1, -2) @
                            np.expand_dims(result.grad, axis=result_axis),
                            axis=r_indices
                        ),
                        newshape=other.grad.shape
                    )

            result._backward = _backward
            result._prev = {self, other}

        return result

    def sum(self) -> "Tensor":
        result = Tensor(np.sum(self.data), requires_grad=self.requires_grad)

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += np.ones_like(self.data) * result.grad

            result._backward = _backward
            result._prev = {self}

        return result

    def __neg__(self):
        return self * -1

    def __sub__(self, other: "Tensor"):
        return self + (-other)

    def __rsub__(self, other: "Tensor"):    # other - self
        return -self + other

    def __radd__(self, other: "Tensor"):  # other + self
        return self + other

    def __rmul__(self, other: "Tensor"):  # other * self
        return self * other

    def __truediv__(self, other: "Tensor"):
        return self * (other ** -1)

    def __rtruediv__(self, other: "Tensor"):    # other / self
        return (self ** -1) * other

    def __repr__(self) -> str:
        if self.requires_grad:
            return f"Tensor({self.data}, requires_grad={self.requires_grad})"

        return f"Tensor({self.data})"


def broadcast(left: tuple, right: tuple) -> (tuple, tuple):
    lsize = len(left)
    rsize = len(right)

    maxsize = max(lsize, rsize)

    newleft = (1,) * (maxsize - lsize) + left
    newright = (1,) * (maxsize - rsize) + right

    assert len(newleft) == len(newright)

    # now we can collect the indices where right or left needs to broadcast to match the dim
    l_indices, r_indicies = [], []
    for i in range(len(newleft)):
        if newleft[i] > newright[i]:
            r_indicies.append(i)
        else:
            l_indices.append(i)
    return tuple(l_indices), tuple(r_indicies)
