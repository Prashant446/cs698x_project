import numpy as np
from numpy.random import randn
from torch.utils.data import Dataset

class Dataclass(Dataset):
	def __init__(self, x_train, y_train=None, transform=None):
		self.data = x_train
		self.targets = y_train
		self.transform = transform

	def __getitem__(self, index):
		y = self.data[index]
		if self.transform is not None:
				y = self.transform(y)
		if self.targets is not None:
				return y, self.targets[index]
		else:
				return y

	def __len__(self):
		return len(self.data)

def load_wiggle():
	np.random.seed(0)
	Npoints = 300
	x = randn(Npoints) * 2.5 + 5  # uniform(0, 10, size=Npoints)

	def function(x):
			return np.sin(np.pi * x) + 0.2 * np.cos(np.pi * x * 4) - 0.3 * x

	y = function(x)

	homo_noise_std = 0.25
	homo_noise = randn(*x.shape) * homo_noise_std
	y = y + homo_noise

	x = x[:, None]
	y = y[:, None]

	x_means, x_stds = x.mean(axis=0), x.std(axis=0)
	y_means, y_stds = y.mean(axis=0), y.std(axis=0)

	X = ((x - x_means) / x_stds).astype(np.float32)
	Y = ((y - y_means) / y_stds).astype(np.float32)

	return X, Y

