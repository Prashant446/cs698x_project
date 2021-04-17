import torch
from torch import nn

class Res_MLPblock(nn.Module):
	def __init__(self, width):
		super(Res_MLPblock, self).__init__()
		self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(num_features=width))
	def forward(self,x):
		return x + self.block(x)