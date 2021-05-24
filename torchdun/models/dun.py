import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from torchdun.models.layers import Res_MLPblock, ConvLSTM


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
	
	def save(self, path):
		args = {
						"input_dim": self.input_dim,
						"hidden_dim": self.hidden_dim,
						"output_dim": self.output_dim,
						"n_layer": self.n_layer
					 }
		
		params = {
				'args': args,
				'state_dict': self.state_dict()
		}
		torch.save(params, path)
	
	@staticmethod
	def load(model_path):
		params = torch.load(model_path, map_location=lambda storage, loc: storage)
		args = params['args']
		model = Dun_fc_res(**args)
		model.load_state_dict(params['state_dict'])
		return model
	


class dun_convlstm(nn.Module):
	def __init__(self, n_layer):
		super(dun_convlstm, self).__init__()
		self.n_layer = n_layer
		self.lstm = ConvLSTM(input_dim=1, hidden_dim=2, kernel_size=(3,3), 
													num_layers=3, bias=False, return_all_layers=True, 
													batch_first=True)


		self.output_block = nn.Sequential( nn.Dropout2d(0.5) 
																				, nn.Flatten()
																				, nn.Linear(500, 16)
																				, nn.ReLU()
																				, nn.Linear(16, 1)
																			)
	def forward(self, x):
		out, states = self.lstm(x)
		last_states = states[-1][0]
		last_states = self.output_block(last_states)
		return last_states
	
	def save(self, path):
		args = {
						# "input_dim": self.input_dim,
						# "hidden_dim": self.hidden_dim,
						# "output_dim": self.output_dim,
						"n_layer": self.n_layer
					 }
		
		params = {
				'args': args,
				'state_dict': self.state_dict()
		}
		torch.save(params, path)
	
	@staticmethod
	def load(model_path):
		params = torch.load(model_path, map_location=lambda storage, loc: storage)
		args = params['args']
		model = dun_convlstm(**args)
		model.load_state_dict(params['state_dict'])
		return model