import numpy


class Preprocessing(object):
	
	def __call__(self):
		raise NotImplementedError


class Dataset(object):

	def __getitem__(self, index):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError


class Sampler(object):

	def __iter__(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError


class StandardSampler(Sampler):

	def __init__(self, dataset, shuffle, batchsize, droplast):
		self.dataset = dataset
		if shuffle:
			self.indexes = iter(numpy.random.permutation(len(dataset)))
		else:
			self.indexes = iter(numpy.arange(len(dataset)))
		self.batchsize = batchsize
		self.droplast = droplast

	def __iter__(self):
		return next(self)

	def __next__(self):
		indexes = []
		for index in self.indexes:
			indexes.append(index)
			if len(indexes) == self.batchsize:
				yield [self.dataset[index] for index in indexes]
				indexes = []
		if len(indexes) > 0 and not self.droplast:
			yield [self.dataset[index] for index in indexes]

	def __len__(self):
		if self.droplast:
			return len(self.dataset) // self.batchsize
		else:
			return (len(self.dataset) + self.batchsize - 1) // self.batchsize
