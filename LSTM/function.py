import numpy


def sigmoid(x):
    return 1. / (1. + numpy.exp(-x))


def d_sigmoid(x):
    return numpy.exp(-x) / ((1. + numpy.exp(-x)) ** 2)


def tanh(x):
    return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))


def d_tanh(x):
    return 1. - tanh(x) ** 2


def relu(x):
    return x * (x > 0.)


def d_relu(x):
    return 1. * (x > 0.)


def softplus(x):
    return numpy.log(1. + numpy.exp(x))


def d_softplus(x):
    return numpy.exp(x) / (1. + numpy.exp(x))


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x))


def negative_log(probability, label):
    return numpy.sum(-numpy.log(probability) * label)


def d_negative_log(probability, label):
    return probability - label


def one_hot_encoding(vectorsize, hot):
    return numpy.array([(lambda x: 1. if x == hot else 0.)(index) for index in range(vectorsize)])


def maximum_index(vector, axis=None):
    return numpy.argmax(vector, axis=axis)
