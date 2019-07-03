import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from data.datasets import EpisodeDataset

parser = argparse.ArgumentParser()
parser.add_argument('root', help='Root directory containing images splits and classes.txt')

args = parser.parse_args()

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor()
])


class ConvolutionalBlock(nn.Module):
	def __init__(self, n_input, size=64):
		super().__init__()

		self.conv = nn.Conv2d(n_input, size, 3, padding=1)
		self.batch_norm = nn.BatchNorm2d(size)
		self.pooling = nn.MaxPool2d(2)

	def forward(self, x):
		x = self.conv(x)
		x = self.batch_norm(x)
		x = F.relu(x)
		return self.pooling(x)


class PrototypicalNetwork(nn.Module):
	def __init__(self, n_input, num_blocks=4, block_size=64):
		super().__init__()

		self.__n_input = n_input
		self.__block_size = block_size

		blocks = list(map(self.__block_mapper, range(num_blocks)))
		self.__body = nn.Sequential(*blocks)

	def __block_mapper(self, i):
		n_input = self.__block_size
		if i == 0:
			n_input = self.__n_input

		return ConvolutionalBlock(n_input, self.__block_size)

	def forward(self, x):
		y = self.__body(x)
		return y.view(x.size(0), -1)

model = PrototypicalNetwork(3)

dataset = EpisodeDataset(args.root, 1, 5, transform=transform)
loader = DataLoader(dataset)

for x, y in loader:
	pass
