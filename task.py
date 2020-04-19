import torch
import torch.nn as nn

class Task:
	def __init__(self, v_size):
		self.v_size = v_size
		self.criterion = nn.CrossEntropyLoss()

	def train_step(self, model, batch):
		model.zero_grad()
		loss = self.criterion(model(batch).view(-1, self.v_size), batch['trg'].view(-1))
		loss.backward()
		return loss.item()

