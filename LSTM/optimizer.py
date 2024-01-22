import numpy


class Optimizer(object):

    def __call__(self):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate=lambda t: 0.001, parameter=None):
        self.learning_rate = learning_rate
        self.parameter = parameter
        self.t = 0

    def __call__(self, gradient):
        self.t += 1
        for key in self.parameter.keys():
            self.parameter[key] -= self.learning_rate(self.t) * gradient[key]
        return self.parameter


class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameter=None):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter = parameter
        self.m = {key: 0. for key in parameter.keys()}
        self.v = {key: 0. for key in parameter.keys()}
        self.t = 0

    def __call__(self, gradient):
        self.t += 1
        for key in self.parameter.keys():
            self.m[key] = self.beta1 * self.m[key] + (1. - self.beta1) * gradient[key]
            self.v[key] = self.beta2 * self.v[key] + (1. - self.beta2) * gradient[key] ** 2
            corrected_m = self.m[key] / (1. - self.beta1 ** self.t)
            corrected_v = self.v[key] / (1. - self.beta2 ** self.t)
            self.parameter[key] -= self.alpha * corrected_m / (numpy.sqrt(corrected_v) + self.epsilon)
        return self.parameter


class AdaMax(Optimizer):

    def __init__(self, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, parameter=None):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter = parameter
        self.m = {key: 0. for key in parameter.keys()}
        self.u = {key: 0. for key in parameter.keys()}
        self.t = 0

    def __call__(self, gradient):
        self.t += 1
        for key in self.parameter.keys():
            self.m[key] = self.beta1 * self.m[key] + (1. - self.beta1) * gradient[key]
            self.u[key] = numpy.maximum(self.beta2 * self.u[key], numpy.abs(gradient[key]))
            corrected_m = self.m[key] / (1. - self.beta1 ** self.t)
            self.parameter[key] -= self.alpha * corrected_m / (self.u[key] + self.epsilon)
        return self.parameter
