import gzip
import pickle
import numpy as np


def relu(x):
    return x * (x > 0.)


def d_relu(x):
    return 1. * (x > 0.)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def negative_log(probability, label):
    return np.sum(-np.log(probability) * label)


def d_negative_log(probability, label):
    return probability - label


class MLP:
    
    def __init__(self):
        self.layer = dict(x=None, h=None, y=None)
        self.parameter = dict(wh=None, bh=None, wy=None, by=None)
        self.gradient = dict(x=None, h=None, y=None, wh=None, bh=None, wy=None, by=None)

    def randomize(self):
        self.parameter['wh'] = np.random.uniform(-0.08, 0.08, (128, 784))
        self.parameter['bh'] = np.random.uniform(-0.08, 0.08, (128, ))
        self.parameter['wy'] = np.random.uniform(-0.20, 0.20, (10, 128))
        self.parameter['by'] = np.random.uniform(-0.20, 0.20, (10, ))

    def forward(self, x):
        self.layer['x'] = x
        self.layer['h'] = np.dot(self.parameter['wh'], self.layer['x']) + self.parameter['bh']
        self.layer['y'] = np.dot(self.parameter['wy'], relu(self.layer['h'])) + self.parameter['by']
        return softmax(self.layer['y'])

    def loss(self, probability, label):
        return negative_log(probability, label)

    def backward(self, probability, label):
        self.gradient['y'] = d_negative_log(probability, label)
        self.gradient['h'] = np.dot(self.gradient['y'], self.parameter['wy']) * d_relu(self.layer['h'])
        self.gradient['by'] = self.gradient['y']
        self.gradient['wy'] = self.gradient['y'].reshape(-1, 1) * relu(self.layer['h'])
        self.gradient['bh'] = self.gradient['h']
        self.gradient['wh'] = self.gradient['h'].reshape(-1, 1) * self.layer['x']
        return self.gradient


class SGD:

    def __init__(self, parameter):
        self.parameter = parameter

    def optimize(self, gradient):
        for name in self.parameter.keys():
            self.parameter[name] -= 0.001 * gradient[name]
        return self.parameter


if __name__ == '__main__':


	MLP = MLP()
	MLP.randomize()

	
	SGD = SGD(MLP.parameter)
	

	with gzip.open('dataset/MNIST.train.pkl.gz', 'rb') as f:
		images, labels = pickle.load(f)


	for epoch in range(1):
		print('Epoch {:d}'.format(epoch))
		loss = []
		for index in np.random.permutation(60000):
			image, label = images[index], labels[index]
			probability = MLP.forward(image)
			loss.append(MLP.loss(probability, label))
			gradient = MLP.backward(probability, label)
			MLP.parameter = SGD.optimize(gradient)
		print('Average Loss: {:.5f}'.format(np.mean(loss)))


	with gzip.open('dataset/MNIST.test.pkl.gz', 'rb') as f:
		images, labels = pickle.load(f)


	correct = 0
	for index in range(10000):
		image, label = images[index], labels[index]
		probability = MLP.forward(image)
		if np.argmax(probability) == np.argmax(label):
			correct += 1
	print('Accuracy Rate: {:.2%}'.format(correct / 10000))
