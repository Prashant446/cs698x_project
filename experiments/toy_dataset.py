import numpy as np
import torch
from torchdun.models.dun import Dun_fc_res
from torchdun.probability import depth_categorical_VI
from torchdun.data import Dataclass, load_agw_1d, load_andrew_1d, load_axis, load_matern_1d, load_my_1d, load_origin, load_wiggle
import matplotlib.pyplot as plt

prefix = "matern_1d/"
load_data = {
	"wiggle": load_wiggle,
	"andrew_1d": load_andrew_1d,
	"agw_1d": load_agw_1d,
	"axis": load_axis,
	"simple_1d": load_my_1d,
	"origin": load_origin,
	"matern_1d": load_matern_1d
}
# wiggle synthetic data
if prefix[:-1] in ["simple_1d"]:
	X, Y, _, _ = load_data[prefix[:-1]]("./data")
else:
	X, Y = load_data[prefix[:-1]]("./data")

print(X.shape, Y.shape)
plt.figure(dpi=80)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.tight_layout()
plt.scatter(X.flatten(),Y.flatten(), s=5, alpha=0.5,)
plt.title(f'{prefix[:-1]} dataset')
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(prefix+"data.png")


trainset = Dataclass(X,Y)
valset = Dataclass(X,Y)

batch_size = len(trainset)
n_cpu = 10
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                          num_workers=n_cpu)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=n_cpu)

n_layer = 10
N = len(trainset)
# Depth: 0 to n_layer => (n_layer+1) possiblities
prior = [1/(n_layer+1)]*(n_layer+1) # uniform prior
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Dun_fc_res(input_dim=X.shape[1], hidden_dim=100, output_dim=Y.shape[1], n_layer=n_layer).to(device)
vi = depth_categorical_VI(prior, N).to(device)
params = list(model.parameters()) + list(vi.parameters())
optim = torch.optim.SGD(params,lr=1e-3,momentum = 0.9, weight_decay=0.0001)

n_epoch = 4000
run_elbo = []
run_depth_prob = []
log_elbo = []
log_depth_prob = []
log_every = 10

for epoch in range(n_epoch):
	for itr,(x,y) in enumerate(trainloader):
		vi.update_prior()
		x = x.to(device)
		y = y.to(device)
		act_vec = model(x)
		neg_elbo = -vi.ELBO(y, act_vec)/N
		elbo = -neg_elbo.item()*N
		optim.zero_grad()
		neg_elbo.backward()
		optim.step()
		run_elbo.append(elbo)
		log_elbo.append(elbo)
		run_depth_prob.append(vi.last_post.cpu().numpy())
		log_depth_prob.append(vi.last_post.cpu().numpy())
		# print(vi.beta.cpu().numpy())
		if epoch%log_every == 0:
			print(f'epoch:{epoch+1}, itr:{itr+1} avg. elbo:{np.mean(log_elbo)}')
			print(f'depth prob: {np.mean(log_depth_prob, axis=0)}')
			log_elbo = []
			log_depth_prob = []

# SAVE model
model.save(prefix+"model.pth")


# TODO: use loop
run_depth_prob = np.array(run_depth_prob)
fig, ax = plt.subplots()
ax.plot(range(run_depth_prob.shape[0]),run_depth_prob[:,0], label='d=0')
ax.plot(range(run_depth_prob.shape[0]),run_depth_prob[:,1], label='d=1')
ax.plot(range(run_depth_prob.shape[0]),run_depth_prob[:,2], label='d=2')
ax.plot(range(run_depth_prob.shape[0]),run_depth_prob[:,3], label='d=3')
ax.plot(range(run_depth_prob.shape[0]),run_depth_prob[:,4], label='d=4')
ax.plot(range(run_depth_prob.shape[0]),run_depth_prob[:,5], label='d=5')
ax.set(ylim=(10e-4,1))
ax.set_yscale('log')
ax.set_yticks([10e-3, 10e-2, 10e-1])
plt.xlabel("$epochs$")
plt.ylabel("$p(d)$")
plt.title("Depth Posterior")
ax.legend(loc="upper left")
plt.savefig(prefix+'posterior.png')

print(f'Learned Likelihood var: {vi.fNLL.log_std.exp()**2}')

def plot_pred(name, title, x, y, pred_mean, pred_var=None):
	plt.figure(dpi=80)
	plt.xlim(-4,4)
	plt.ylim(-4,4)
	# plt.tight_layout()
	plt.scatter(x,y, s=5, alpha=0.5,)
	plt.plot(x_test,pred_mean)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title(title)
	if pred_var is not None:
		plt.fill_between(x_test.squeeze(), pred_mean.squeeze() - pred_var.squeeze(), 
									pred_mean.squeeze()+pred_var.squeeze(),alpha=0.3)
	plt.savefig(prefix + name)


model = model.load(prefix+"model.pth").to(device)
model.eval()
vi.eval()

for itr,(x,y) in enumerate(trainloader):
	x = x.to(device)
	y = y.to(device)
	with torch.no_grad():
		x_test = torch.arange(torch.min(x)-2, torch.max(x)+1, step=0.02, device = device).view(-1,1)
		act_vec = model(x_test)

		pred_mean, pred_var = vi.ppd(act_vec)
		pred_mean = pred_mean.cpu().numpy()
		pred_var = pred_var +  vi.fNLL.log_std.exp()**2
		pred_var = np.power(pred_var.cpu().numpy(),0.5)
		x = x.cpu().numpy()
		x_test = x_test.cpu().numpy()
		y = y.cpu().numpy() 
		for d, d_act in enumerate(act_vec):
			plot_pred(f'pred_d_f{d}.png',f'd = {d}',x,y,d_act.cpu().numpy())
		plot_pred("pred.png", "Predictions", x,y,pred_mean, pred_var)
