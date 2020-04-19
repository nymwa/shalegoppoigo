import os
import torch
from logging import getLogger
logger = getLogger(__name__)

class Trainer:
	def __init__(self, model, loader, task, optimizer, max_iter, device = None):
		self.model = model.to(device)
		self.train_loader = loader
		self.task = task
		self.optimizer = optimizer
		self.max_iter = max_iter
		self.device = device

	def send(self, batch):
		for key in batch:
			batch[key] = batch[key].to(self.device)
		return batch

	def save(self, epoch):
		os.makedirs('checkpoints', exist_ok=True)
		torch.save(self.model.state_dict(), 'checkpoints/checkpoint{}.pt'.format(epoch))

	def train_epoch(self):
		self.model.train()
		acc = 0
		for n, batch in enumerate(self.train_loader):
			batch = self.send(batch)
			acc += self.task.train_step(self.model, batch)
			self.optimizer.step()
		return acc / n 

	def train(self):
		for epoch in range(self.max_iter):
			train_loss = self.train_epoch()
			self.save(epoch)
			logger.info('epoch {}, train_loss:{:.5f}'.format(epoch, train_loss))

