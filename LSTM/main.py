import time
import gzip
import pickle
import numpy
import dataset
import function
import optimizer
import model


def load(path):
	with gzip.open(path, 'rb') as f:
		data = pickle.load(f)
	return data


def save(data, path):
	with gzip.open(path, 'wb') as f:
		pickle.dump(data, f)


class Preprocessing(dataset.Preprocessing):

	def __call__(self, input, label):
		return input.reshape(28, 28), function.one_hot_encoding(vectorsize=10, hot=label)


class Dataset(dataset.Dataset):
	
	def __init__(self, path, optional, preprocessing):
		trainset = load(path)
		if optional is 'train':
			self.inputs, self.labels = trainset
		elif optional is 'test':
			self.inputs, self.labels = testset
		else:
			raise ValueError('optional is train or test')
		self.preprocessing = preprocessing

	def __getitem__(self, index):
		return self.preprocessing(self.inputs[index], self.labels[index])

	def __len__(self):
		return len(self.inputs)		


if __name__ == '__main__':

	
	# training
	trainset = Dataset(path='dataset/MNIST.train.pkl.gz', optional='train', preprocessing=Preprocessing())
	model = model.LSTM(inputsize=28, hiddensize=128, outputsize=10)
	initial_s, initial_h = numpy.zeros((128, )), numpy.zeros((128, ))
	# model.parameter = load(path='parameter/epoch[0].pkl.gz')
	optimizer = optimizer.Adam(parameter=model.parameter)


	for epoch_number in range(1, 10):
		print('Epoch {:d}:'.format(epoch_number))
		error_sum = 0.
		sampler = dataset.StandardSampler(dataset=trainset, shuffle=True, batchsize=20, droplast=True)
		start = time.time()
		for batch_number, batch in enumerate(sampler, start=1):
			gradient_sum = {key: 0. for key in model.parameter.keys()}
			for sample_number, sample in enumerate(batch, start=1):
				input, label = sample
				probability = model.forward(initial_s, initial_h, input)
				error = model.error(probability, label)
				error_sum += error
				gradient = model.backward(probability, label)
				gradient_sum = {key: gradient_sum[key] + gradient[key] for key in model.parameter.keys()}
			gradient_average = {key: gradient_sum[key] / sample_number for key in model.parameter.keys()}
			model.parameter = optimizer(gradient_average)
			print('Progress: {:.2%}'.format(batch_number / len(sampler)), end='\r')
		print('Average Error: {:.5f} Time: {:.2f}'.format(error_sum / len(trainset), time.time() - start))
		save(model.parameter, path='parameter/epoch[{:d}].pkl.gz'.format(epoch_number))
	
	'''
	# testing
	testset = Dataset(path='dataset/mnist.pkl.gz', optional='test', preprocessing=Preprocessing())
	model = model.MLP(inputsize=784, hiddensize=128, outputsize=10)
	model.parameter = load(path='parameter/epoch[3].pkl.gz')

	
	correct_number = 0
	sampler = dataset.StandardSampler(dataset=testset, shuffle=False, batchsize=1, droplast=False)
	start = time.time()
	for batch_number, batch in enumerate(sampler, start=1):
		for sample_number, sample in enumerate(batch, start=1):
			input, label = sample
			probability = model.forward(input)
			if function.maximum_index(probability) == function.maximum_index(label):
				correct_number += 1
		print('Progress: {:.2%}'.format(batch_number / len(sampler)), end='\r')
	print('Accuracy Rate: {:.2%} Time: {:.2f}'.format(correct_number / len(testset), time.time() - start))
	'''