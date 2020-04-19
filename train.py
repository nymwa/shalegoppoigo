from dataset import Dataset
from loader import Loader
from trainer import Trainer
from task import Task
from rnn import RNN
from utils import load_vocab, set_seed
import torch
from torch import optim
import sys
import logging
from logging import getLogger, Formatter, StreamHandler
logger = getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG, stream=sys.stdout)

if __name__ == '__main__':
	set_seed(0)
	vlist, vdict = load_vocab()
	dataset = Dataset('train.txt')
	loader = Loader(dataset, 2000)
	model = RNN(len(vlist), 8, 64)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	trainer = Trainer(model, loader, Task(len(vlist)), optimizer, 1000, torch.device('cuda'))
	trainer.train()

