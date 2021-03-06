import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
	def __init__(self, lr, input_dims, fc1_dim, fc2_dim, n_actions, checkpoint_path, name):
		super(DQN, self).__init__()

		self.input_dims = input_dims
		self.fc1_dim = fc1_dim
		self.fc2_dim = fc2_dim
		self.n_actions = n_actions
		
		self.checkpoint_path = checkpoint_path
		self.name = name
		
		self.fc1 = nn.Linear(self.input_dims, self.fc1_dim)
		self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
		self.fc3 = nn.Linear(self.fc2_dim, self.n_actions)

		self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self,state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)								# Keep predicted Q-values raw	(-inf, +inf)
		return x

	def save_checkpoint(self, i, score):
		print('... saving ckpt ...')
		model_path = f'{self.checkpoint_path}/{self.name}_{i}_Weights_score_{score}.pt'
		T.save(self.state_dict(), model_path)

	def load_checkpoint(self, model_path):
		print('... loading ckpt ...')
		self.load_state_dict(T.load(model_path, map_location=self.device))
	
	def __repr__(self):
		return f'Fully Connected Neural Network: {self.name} \n \
						Dimentions = {self.input_dims} | {self.fc1_dim} | {self.fc2_dim} | {self.n_actions} \n \
						Optimizer  = {self.optimizer} \n \
						Loss Function = {self.loss}'

class Dueling_DQN(DQN):
	def __init__(self, lr, input_dims, fc1_dim, fc2_dim, fc3_dim,  n_actions, checkpoint_path, name):
		super().__init__(lr, input_dims, fc1_dim, fc2_dim, n_actions, checkpoint_path, name)

		self.fc3_dim = fc3_dim
		
		self.fc1 = nn.Linear(self.input_dims, self.fc1_dim)
		self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)

		self.fc3_advantage = nn.Linear(self.fc2_dim, self.fc3_dim)
		self.fc3_value = nn.Linear(self.fc2_dim, self.fc3_dim)

		self.fc4_advantage = nn.Linear(self.fc3_dim, self.n_actions)
		self.fc4_value = nn.Linear(self.fc3_dim, 1)

	def forward(self,state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))

		x_advantage = F.relu(self.fc3_advantage(x))
		x_advantage = self.fc4_advantage(x_advantage)

		x_values = F.relu(self.fc3_value(x))
		x_values = self.fc4_value(x_values)

		qvals = x_values + (x_advantage - x_advantage.mean())
		return qvals
