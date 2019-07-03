from torch import nn
from model.block import ConvolutionalBlock

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
