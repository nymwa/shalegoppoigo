from utils import load_vocab
import torch
from torch.nn.utils.rnn import pad_sequence as pad

class Dataset:
	def __init__(self, filepath):
		vlist, vdict = load_vocab()
		with open(filepath) as f:
			 data = f.read().splitlines()
		data = [sent.split(' ') for sent in data]
		self.data = [[vdict[x] for x in sent] for sent in data]
		self.lengths = torch.tensor([len(x) + 1 for x in self.data])
		self.size = len(self.lengths)

	def __len__(self):
		return self.size

	def __call__(self, indices):
		return {
				'src':pad([torch.tensor([0] + self.data[index]) for index in indices]),
				'trg':pad([torch.tensor(self.data[index] + [0]) for index in indices], padding_value=-100),
				'len':self.lengths[indices],
			}

