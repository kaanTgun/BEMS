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
		self.fc3(x)								# Keep predicted Q-values raw	(-inf, +inf)
		return x

	def save_checkpoint(self, loss):
		print('... saving ckpt ...')
		model_path = f'{self.checkpoint_path}/{self.name}_Weights_loss_{loss}.pt'
		T.save(self.state_dict(), model_path)

	def load_checkpoint(self, file_name):
		print('... loading ckpt ...')
		model_path = f'{self.checkpoint_path}/{file_name}'
		self.load_state_dict(T.load(model_path, map_location=self.device))

class LSTM(nn.Module):
	def __init__(self, input_dims, output_dims, hidden_dims=2):
		super(LSTM, self).__init__()

		self.input_dims = input_dims
		self.output_dims=output_dims
		
		self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers=2, batch_first=True)
		# The linear layer to map hidden layers to output using linear transformation
		self.output = nn.Linear(self.input_dims, self.output_dims)

		self.optimizer = T.optim.Adam(self.parameters(), lr=0.05)
		self.loss = nn.MSELoss()
		
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, norm_sequance):
		lstm_out = self.lstm(norm_sequance)
		output = self.output(lstm_out)
		x = F.softmax(output)
		
		return x