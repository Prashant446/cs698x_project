import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .layers import Res_MLPblock


class Dun_fc_res(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
		super(Dun_fc_res, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.n_layer = n_layer
		self.input_layer = nn.Linear(input_dim, hidden_dim)
		self.output_layer = nn.Linear(hidden_dim, output_dim)
		self.hidden_blocks = nn.ModuleList()
		for _ in range(n_layer):
			self.hidden_blocks.append(Res_MLPblock(hidden_dim))

	def forward(self, x, depth=None):
		r"""
		x: (batch_size, input_dim)
		depth: number of hidden blocks to evaluate
		returns act_vec: (D+1, batch_size, output_dim)
		"""
		depth = self.n_layer if depth is None else depth
		assert depth <= self.n_layer
		# (D+1, batch_size, output_dim)
		act_vec = torch.zeros(
				depth+1, x.shape[0], self.output_dim).type(x.type())
		x = self.input_layer(x)
		act_vec[0] = self.output_layer(x)
		for i in range(depth):
				x = self.hidden_blocks[i](x)
				act_vec[i+1] = self.output_layer(x)
		return act_vec



