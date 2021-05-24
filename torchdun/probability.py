import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class GaussianNLL(nn.Module):
	def __init__(self, input_dim, std=None, eps=1e-6):
		super(GaussianNLL, self).__init__()
		if std is None:
			self.log_std = nn.Parameter(torch.zeros(input_dim))
		else:
		# freezed std
			self.log_std = nn.Parameter(
				torch.ones(input_dim) * np.log(std), 
				requires_grad=False)
		self.eps = eps

	def forward(self, mean, input):
		std = self.log_std.exp().clamp(min=self.eps)
		var = std**2
		# sanity check
		assert input.shape == mean.shape
		N = input.shape[0]
		# if var.shape[1:] != 1:
		#     assert input.shape == var.shape
		# else:
		#     assert var.shape[0] == N

		data_shape = input.shape[1:]
		# flatten
		input = input.view(N, -1)
		mean = mean.view(N, -1)
		# var = var.view(N,-1)
		# print(f'input:{input}, mean:{mean}, var:{torch.maximum(var,eps)}')
		NLL = 0.5*torch.sum(
												torch.log(2*torch.tensor(np.pi)) + torch.log(var) + (input-mean)**2 /var
												, dim=1
											)
		# revert to original shape
		NLL = NLL.view(N, *data_shape)
		return NLL


class depth_categorical_VI(nn.Module):
	def __init__(self, prior_prob, N, eps=1e-33):
		super(depth_categorical_VI, self).__init__()
		self.N = N
		self.eps = eps
		self.depth = len(prior_prob)
		self.register_buffer('beta', torch.tensor(prior_prob))
		self.register_buffer('log_beta', torch.log(self.beta))
		# valid prior
		assert self.beta.sum().item() - 1 < 1e-6
		# use logits to avoid constraints
		self.alpha_logits = nn.parameter.Parameter(
			torch.zeros(self.depth), requires_grad=True)
		self.last_post = None
		self.fNLL = GaussianNLL(1)
	
	@staticmethod
	def get_probs(logits):
		return F.softmax(logits, dim=0)
	
	@staticmethod
	def get_logprobs(logits):
		return F.log_softmax(logits, dim=0)

	def get_KL(self):
		log_alpha = self.get_logprobs(self.alpha_logits)
		log_beta = self.log_beta
		# if any alpha collapse to 0 then it will not be updated anymore
		alpha = self.get_probs(self.alpha_logits).clamp(min=self.eps, max=(1 - self.eps))
		kl_div =  torch.sum(alpha*(log_alpha - log_beta))
		return kl_div
	
	def ppd(self, act_vec):
		"""	act_vec (D, batch_size, *output_dim): scores
				returns mean (batch_size, *output_dims): prediction means
				returns var (batch_size, *output_dims): prediction means
		"""
		alpha = self.get_probs(self.alpha_logits)
		proj_shape = (1,)*(len(act_vec.shape)-1) # (1,)*(1 + batch_size)
		alpha_proj = alpha.view(-1, *proj_shape)
		mean = torch.sum(act_vec * alpha_proj, dim=0)
		var = torch.sum((act_vec**2) * alpha_proj, dim=0)
		
		var = var - mean**2 + self.fNLL.log_std.exp()**2
		return mean, var

	def ELBO(self, y, act_vec):
		"""
		y       (batch_size, *output_dims) : label
		act_vec (depth, batch, *output_dim): model scores
		returns 
		"""
		batch_size = act_vec.shape[1]
		# (depth, batch_size)
		loglike_per_act = self.get_depth_wise_LL(y, act_vec)
		alpha = self.get_probs(self.alpha_logits)
		wtd_loglike_per_act = loglike_per_act*torch.unsqueeze(alpha,1)
		expected_LL = self.N*torch.sum(wtd_loglike_per_act)/batch_size
		kl_div =  self.get_KL()
		return expected_LL - kl_div

	def update_prior(self):
		self.last_post = self.get_probs(self.alpha_logits).detach()


	def get_depth_wise_LL(self, y, act_vec):
		"""
		y       (batch_size, *output_dims) : label
		act_vec (depth, batch, *output_dim): scores
		returns ll (depth, batch)
		"""
		depth = act_vec.shape[0]
		batch_size = act_vec.shape[1]
		rep_dim = [depth] + [1 for _ in range(len(y.shape)-1)]
		# (depth*batch_size, *output_dims) with batch changing first
		y_expand = y.repeat(*rep_dim)
		# (depth*batch_size, *output_dims) with batch changing first
		act_vec_flat = act_vec.view(depth*batch_size, -1)
		# TODO: use function here
		loglike_per_act = -self.fNLL(act_vec_flat, y_expand).view(depth, batch_size)
		return loglike_per_act
