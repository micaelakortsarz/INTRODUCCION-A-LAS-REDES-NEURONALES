import numpy as np

class Activation():
    def __call__(self,x): pass
    def gradient(self,x): pass

class ReLu(Activation):
    def __call__(self,x):
        return np.maximum(0, x)

    def gradient(self,x):
        margins = np.maximum(0, x)
        binary = margins.copy()
        binary[binary > 0] = 1
        return binary

class Tanh(Activation):
    def __call__(self,x):
        return np.tanh(x)

    def gradient(self,x):
        return 1-((np.tanh(x))**2)

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def gradient(self,x):
        return np.exp(-1 * x) / (1 + np.exp(-1 * x)) ** 2

class Lineal(Activation):
    def __call__(self,x):
        return x

    def gradient(self,x):
        return 1
