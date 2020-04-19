from sampler import Sampler

class Loader:
	def __init__(self, dataset, max_tokens):
		self.dataset = dataset
		self.sampler = Sampler(dataset, max_tokens)

	def __iter__(self):
		for indices in self.sampler:
			yield self.dataset(indices)

