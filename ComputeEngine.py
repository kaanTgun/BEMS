from Networks import DQN, Dueling_DQN
from Utils import Memory
import Utils

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linprog
import numpy as np
import pandas 
from torch.utils.tensorboard import SummaryWriter

class base_Actor():
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_actions, name, OUTPUT_PATH, eps_end=0.01, eps_dec=5e-4):
		""" Initialte the Actor Agent 

		Args:
				gamma (float): 			Discount rate of max reward
				epsilon (float): 		Randomness
				lr (float): 				Learining rate
				input_dims (int): 		Dims of state space
				batch_size (int): 		Batch size
				num_actions (int): 		Number fo actions can be taken in the environment (discrete)
				max_mem_size (int, optional): 	Defaults to 100000.
				eps_end (float, optional): 			Defaults to 0.01.
				eps_dec (float, optional): 			Epsilon declenation. Defaults to 5e-4.
		"""
		self.name = name
		self.output_path = OUTPUT_PATH
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_end = eps_end
		self.eps_dec = eps_dec  
		self.actions_space = [i for i in range(num_actions)]      		# (Max_charge, Stall, Max_discharge) --> [0,1,2]

		self.memory = Memory(input_dims, batch_size)

		Utils.create_log_folders(NETWORK_NAME=self.name, OUTPUT_PATH=self.output_path)
		self.writer = SummaryWriter(f"{self.output_path}/runs")

	def store_transition(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def epsilon_decay(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

	def choose_action(self):
		raise NotImplementedError
	def sample_memory(self):
		raise NotImplementedError
	def save_ckpt(self, i, score):
		raise NotImplementedError
	def load_ckpt(self, path):
		raise NotImplementedError
	def learn(self):
		raise NotImplementedError
	def __str__(self):
		return 'This is an abstract base class, don\'t impliment it directly !'

class DQN_Actor(base_Actor):
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_actions, OUTPUT_PATH='DQN_Model' ,eps_end=0.01, eps_dec=5e-4, name='DQN'):
		super().__init__(gamma, epsilon, lr, input_dims, batch_size, num_actions,name, OUTPUT_PATH ,eps_end, eps_dec)
		
		self.Q_eval = DQN(lr, input_dims, 256, 256, num_actions, \
											checkpoint_path=self.output_path, name=self.name)  # Policy Object	

	def choose_action(self, observations, test=False):
		""" Given the observations of the state, take an action to maximize the policy or take a random action

		Args:
				observations (List): State Space parameters
		"""
		if test:
			self.Q_eval.eval()
			with T.no_grad():
				state = T.tensor(np.asarray(observations, dtype=np.float32)).to(self.Q_eval.device)
				actions = self.Q_eval(state)
				action = T.argmax(actions).item()
				return action

		if np.random.random() > self.epsilon:
			state = T.tensor(np.asarray(observations, dtype=np.float32)).to(self.Q_eval.device)
			actions = self.Q_eval(state)
			
			# Get the best action given the policy (DQN)
			action = T.argmax(actions).item()
		else:
			action = np.random.choice(self.actions_space)
		
		return action

	def sample_memory(self):
		state, action, reward, new_state, terminal = self.memory.sample_batch()

		state     =T.tensor(state).to(self.Q_eval.device)
		new_state =T.tensor(new_state).to(self.Q_eval.device)
		reward    =T.tensor(reward).to(self.Q_eval.device)
		terminal  =T.tensor(terminal).to(self.Q_eval.device)
		return state, action, reward, new_state, terminal

	def save_ckpt(self, i, score):
		self.Q_eval.save_checkpoint(i, score)
	
	def load_ckpt(self, path):
		self.Q_eval.load_checkpoint(path)

	def learn(self):
		""" To start learing, fill up at least single batch_size of experience
		""" 
		if self.memory.mem_cntr < self.memory.batch_size:
			return
		self.Q_eval.optimizer.zero_grad()

		state, action, reward, new_state, terminal = self.sample_memory()

		batch_index = np.arange(self.memory.batch_size, dtype=np.int32)

		q_eval = self.Q_eval(state)[batch_index, action]    
		# Returns actions values for each batch, so slice by[batch_index, taken_Action] -> Return size(Batch, 1)
		q_next = self.Q_eval(new_state)                          
		# Returns estimated next action values under the current policy 								-> Return size(Batch, num_actions)
		q_next[terminal] = 0.0																						
		# If this is the termination state, set all the state values as 0 							-> no next state 

		q_target = reward + self.gamma * T.max(q_next, dim=1)[0]					
		# Take the max estimated action for each batch, and use Q-Function for calculating q_target
		loss     = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backward()

		self.Q_eval.optimizer.step()

		# Random eplore exploit 
		self.epsilon_decay()

class DoubleDQN_Actor(base_Actor):
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_actions, \
							replace_networks=200 ,eps_end=0.01, eps_dec=2e-4, OUTPUT_PATH='DDQN_Model', name='Double_DQN'):
		super().__init__(gamma, epsilon, lr, input_dims, batch_size, num_actions, name, OUTPUT_PATH ,eps_end, eps_dec)

		self.replace = replace_networks 		
		self.learn_counter = 0

		self.Q_Local = DQN(lr, input_dims, 256, 256, num_actions, \
											checkpoint_path=self.output_path, name="Q_Local")  # Policy Object 1

		self.Q_Target = DQN(lr, input_dims, 256, 256, num_actions, \
											checkpoint_path=self.output_path, name="Q_Target")  # Policy Object 2
	
	def replace_networks(self):
		self.Q_Target.load_state_dict(self.Q_Local.state_dict())

	def choose_action(self, observations, test=False):
		if test:
			self.Q_Local.eval()
			with T.no_grad():
				state = T.tensor(np.asarray(observations, dtype=np.float32)).to(self.Q_Local.device)
				actions = self.Q_Local(state)
				action = T.argmax(actions).item()
				return action

		if np.random.random() > self.epsilon:
			state = T.tensor(np.asarray(observations, dtype=np.float32)).to(self.Q_Local.device)
			actions = self.Q_Local(state)			
			action = T.argmax(actions).item()

		else:
			action = np.random.choice(self.actions_space)		
		return action

	def sample_memory(self):
		state, action, reward, new_state, terminal = self.memory.sample_batch()

		state     =T.tensor(state).to(self.Q_Local.device)
		new_state =T.tensor(new_state).to(self.Q_Local.device)
		reward    =T.tensor(reward).to(self.Q_Local.device)
		terminal  =T.tensor(terminal).to(self.Q_Local.device)
		return state, action, reward, new_state, terminal

	def save_ckpt(self, i, score):
		self.Q_Local.save_checkpoint(i, score)
	
	def load_ckpt(self, path):
		self.Q_Local.load_checkpoint(path)

	def learn(self):
		if self.memory.mem_cntr < self.memory.batch_size:
			return
		
		self.Q_Local.optimizer.zero_grad() 

		if self.learn_counter % self.replace == 0:
			self.replace_networks()

		state, action, reward, new_state, terminal = self.sample_memory()
		batch_index = np.arange(self.memory.batch_size, dtype=np.int32)

		q_local_pred 		= self.Q_Local(state)[batch_index, action]
		q_local_next 		= self.Q_Local(new_state)
		local_max_next_actions = T.argmax(q_local_next, dim=1)

		q_target_next 	= self.Q_Target(new_state)[batch_index, local_max_next_actions]
		q_target_next[terminal] = 0.0
		
		q_estimated = reward + self.gamma * q_target_next

		loss     = self.Q_Local.loss(q_estimated, q_local_pred).to(self.Q_Local.device)
		loss.backward()

		self.Q_Local.optimizer.step()

		self.epsilon_decay()
		self.learn_counter+=1	

class Linear_Programming():
	def __init__(self, DataFile_path, maxCharge, minCharge, rate , batteryCap):
		""" Init linear programming environment

		Args:
				DataFile_path (str): 		csv load data in terms of Cents/kWh
				maxCharge (float): 			Max SOC percent
				minCharge (float): 			Min SOC percent
				rate (float): 					Max rate of charge and discharge of percent power over interval (time)
				batteryCap (int): 			kWh battery capacity
		"""
	
		self.df = pandas.read_csv(DataFile_path)        
		self.max_charge = maxCharge
		self.min_charge = minCharge
		self.rate = rate               
		self.battery_cap = batteryCap

	def linprog_true(self, soc, startIndex, endIndex):
		""" Solve for the optimal strategy using linear programming. 
		Maximise for the profits, given the power price over a period of time.

		Args:
			soc (float): 				Percent State of Charge
			startIndex (int): 	Index of starting data row contains the date and price of power
			endIndex (int): 		Index of ending data row contains the date and price of power

		Returns:
			obj_func (float List): 			Price of power over time
			actions (int List): 				Chage/hold/discharge over time
			soc_overtime (int List): 		Percent soc over time

		"""
		self.start_index = startIndex
		self.end_index   = endIndex
		N = self.end_index - self.start_index

		A = np.append(np.tril(np.ones((N,N))), \
					-np.tril(np.ones((N,N))), 0)
		b = np.append((self.max_charge-soc)*np.ones((N,1)), \
						(soc-self.min_charge)*np.ones((N,1)))
		
		lb = -self.rate
		ub = self.rate

		rowData = self.df.iloc[self.start_index:self.end_index]
		power_price = rowData['HOEP']/1000

		obj_func = np.array(power_price)

		result = linprog(obj_func, A_ub=A, b_ub=b, bounds =[lb,ub])
		actions = np.asarray(result.x)

		soc_overtime = np.matmul(np.transpose(np.expand_dims(actions, axis=1)), np.triu(np.ones((N,N))))

		soc_overtime[0,:] += soc
		soc_overtime = np.sum(soc_overtime, axis=0)

		return obj_func, actions, soc_overtime
	
	def linprog_predict_interval(self, soc, startIndex, endIndex, horizon=24, step=1, noise_magnitude=0.03):
		""" Starting from the startIndex, solve for the optimal strategy for the predicted horizon length (true+noise), 
		take the next action (t+1) and recalculate the predicted horizon again until the endIndex is reached  

		Args:
				soc (float): 								State of charge
				startIndex (int): 					Start index
				endIndex (int): 						End index
				horizon (int, optional): 		Solve for the next n intervals. Defaults to 24.
				step (int, optional): 			Apply linear programming in every given step. Defaults to 1.
				noise_magnitude (float, optional): Applied as the predicttion error. Defaults to 0.03.

		Returns:
				price_overtime (float List): 	Predicted price of power over time
				power_price (float List): 		True price of power over time
				actions (int List): 					Chage/hold/discharge over time
				soc_overtime (int List): 			Percent soc over time
		"""
		self.start_index = startIndex
		self.end_index   = endIndex
		N = self.end_index - self.start_index

		A = np.append(np.tril(np.ones((horizon,horizon))), \
					-np.tril(np.ones((horizon,horizon))), 0)
		b = np.append((self.max_charge-soc)*np.ones((horizon,1)), \
						(soc-self.min_charge)*np.ones((horizon,1)))
		
		lb = -self.rate
		ub = self.rate

		row_Data = self.df.iloc[self.start_index : self.end_index]
		power_price = np.array(row_Data['HOEP']/1000)
		
		soc_overtime = [soc]
		actions = [0]
		price_overtime = [0]

		for i in range(0, N-horizon+1, step):
			noise = np.random.rand(horizon) * noise_magnitude            
			price_with_noise = noise + power_price[i : i+horizon]

			obj_func = np.array(price_with_noise)
			b = np.append((self.max_charge-soc)*np.ones((horizon,1)),\
										(soc-self.min_charge)*np.ones((horizon,1)))
			

			res = linprog(obj_func, A_ub=A, b_ub=b, bounds=[lb,ub])
			action_t = round(100*res.x[0])/100    		# Take the first action computed
			soc += action_t

			actions.append(action_t)
			soc_overtime.append(float(soc))
			price_overtime.append(obj_func[0])

		return np.asarray(price_overtime), np.asarray(power_price), np.asarray(actions), np.asarray(soc_overtime)

	# Enve = Enve(max_charge = 0.8,min_charge = 0.2,rate = 0.1, battery_cap = 1500)
	# power_price, actions, soc_overtime = Enve.Linprog_True(0.6, 1000,1030)
	# t = [i for i in range(len(power_price))]

	# plt.plot(t, soc_overtime, 'b--')
	# plt.plot(t, power_price, 'r--')
	# plt.show

	# price_overtime, power_price, actions, soc_overtime = Enve.Linprog_predict_interval(0.6,1000,1030+24)
	# t = [i for i in range(len(soc_overtime))]
	# plt.plot(t, soc_overtime, 'g--')
	# plt.plot(t, price_overtime, 'p--')
	# plt.show



