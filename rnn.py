import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class RNN(nn.Module):
	def __init__(self, v_size, e_size, h_size):
		super().__init__()
		self.embed = nn.Embedding(v_size, e_size)
		self.rnn = nn.LSTM(e_size, h_size)
		self.out = nn.Linear(h_size, v_size)
		self.dropout = nn.Dropout(0.2)

	def forward(self, batch, h=None):
		x = self.embed(batch['src'])
		x = pack(x, batch['len'])
		x, _ = self.rnn(x, h)
		x, _ = unpack(x)
		x = self.dropout(x)
		x = self.out(x)
		return x

