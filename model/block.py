from torch import nn
import torch.nn.functional as F

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
