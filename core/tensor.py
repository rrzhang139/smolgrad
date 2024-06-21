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

    def __add__(self, other):
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

    def __mul__(self, other):
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

    def sum(self):
        # TODO: Implement sum operation
        pass
