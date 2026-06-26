import math
import random
import logging

logger = logging.getLogger(__name__)


class Value:
    def __init__(self, data, _children=(), op=(), label=" "):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op
        self._label = label

    def __repr__(self):
        return f"Value(label={self._label}, data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only int and float powers are supported for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __gt__(self, other):
        return self.data > other.data

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), label="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), label="log")

        def _backward():
            self.grad += 1 / self.data * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), label="exp")

        def _backward():
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        self._backward()
        # print(f"label: {self._label}, grad: {self.grad}")
        for child in self._prev:
            child.backward()


class Module:
    def parameters(self): ...

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for i in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.w), (
            "number of inputs is not equal to the number of weights"
        )
        act = sum([xi * self.w[i] for i, xi in enumerate(x)]) + self.b
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for la in self.layers for p in la.parameters()]

    def train(self, xs, ys, iter, step):
        for _ in range(iter):
            # forward
            ypreds = [self(x) for x in xs]
            loss = sum([(ypred - y) ** 2 for y, ypred in zip(ys, ypreds)])
            logger.info(f"loss {loss}")
            loss._label = "loss"
            loss.grad = 1
            self.zero_grad()
            # backward
            loss.backward()
            # update
            for p in self.parameters():
                p.data += p.grad * (-step)
