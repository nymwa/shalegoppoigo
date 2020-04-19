import random as rd
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed):
	rd.seed(seed) # for all seeds in random
	np.random.seed(seed) # for all seeds in numpy 
	torch.manual_seed(seed) # for all seeds in pytorch

def load_vocab():
	with open('vocab.txt') as f:
		vlist = f.read().splitlines()
	vdict = {x : i for i, x in enumerate(vlist)}
	return vlist, vdict

def embedding(v_size, e_size):
	x = nn.Embedding(v_size, e_size)
	nn.init.normal_(x.weight, 0, 0.1)
	return x

def layernorm(size):
	x = nn.LayerNorm(size, eps=1e-12)
	x.bias.data.zero_()
	x.weight.data.fill_(1.0)
	return x

def linear(i_size, o_size, bias=True):
	x = nn.Linear(i_size, o_size, bias)
	nn.init.xavier_uniform_(x.weight)
	if bias:
		nn.init.constant_(x.bias, 0.0)
	return x

