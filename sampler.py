import random as rd
import torch

class Sampler:
	def __init__(self, dataset, max_tokens):
		self.dataset = dataset
		self.max_tokens = max_tokens

	def generate_batches(self):
		batches = []
		batch = []
		acc = 0
		max_len = 0
		for index in self.indices:
			acc += 1
			this_len = self.dataset.lengths[index]
			max_len = max(max_len, this_len)
			if acc * max_len > self.max_tokens:
				batches.append(batch)
				batch = [index]
				acc = 1
				max_len = this_len
			else:
				batch.append(index)
		if batch != []:
			batches.append(batch)
		rd.shuffle(batches)
		return batches

	def __iter__(self):
		self.indices = torch.randperm(len(self.dataset))
		self.indices = self.indices[self.dataset.lengths[self.indices].argsort(descending=True)]
		for batch in self.generate_batches():
			yield torch.tensor(batch)

