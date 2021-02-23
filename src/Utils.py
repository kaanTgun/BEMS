import numpy as np
import os

def create_log_folders(NETWORK_NAME, OUTPUT_PATH, LOG_LOSS_DIR="Loss",\
											 LOG_LOSS_T_PATH="Loss/Train", LOG_LOSS_V_PATH="Loss/Validate"):
	if not os.path.exists(OUTPUT_PATH):
		os.mkdir(OUTPUT_PATH)
	# if not os.path.exists(LOG_LOSS_DIR):
	# 	os.mkdir(LOG_LOSS_DIR)
	# 	os.mkdir(LOG_LOSS_T_PATH)
	# 	os.mkdir(LOG_LOSS_V_PATH)

class Memory():
	def __init__(self, input_dims, batch_size, max_mem_size=100000):
		self.mem_size = max_mem_size
		self.batch_size = batch_size

		# Record every taken State Action and Reward observed by appying a certain policy
		self.state_mem 			= np.zeros((self.mem_size, input_dims), dtype=np.float32)
		self.new_state_mem 	= np.zeros((self.mem_size, input_dims), dtype=np.float32)

		self.action_mem 		= np.zeros(self.mem_size, dtype=np.int32)
		self.reward_mem 		= np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_mem 	= np.zeros(self.mem_size, dtype=np.bool)

		# Memory counter for placing the value in the correct location
		self.mem_cntr = 0
	
	def store_transition(self, state, action, reward, new_state, done):
		""" Store the SARS values in memory for experiance replay

		Args:
				state (List): 					List of np.float32 of current state
				action (np.int32): 			Taken action
				reward (np.float32): 		Reward gained from the transition from state1-->state2
				newState (List): 				List of np.float32 of current new state
				done (np.bool): 				If the episode is over or not
		"""
		# Override the memory on new observations
		index = self.mem_cntr % self.mem_size 
		self.state_mem[index] = np.asarray(state)
		self.action_mem[index] = action
		self.new_state_mem[index] = np.asarray(new_state)
		
		self.reward_mem[index] = reward
		self.terminal_mem[index] = done

		self.mem_cntr +=1
	
	def sample_batch(self):
		max_mem     = min(self.mem_cntr, self.mem_size)

		# Generate random indices from trainig dataset
		batch       = np.random.choice(max_mem, self.batch_size, replace=False)

		state_batch     = self.state_mem[batch]     
		new_state_batch = self.new_state_mem[batch]
		reward_batch    = self.reward_mem[batch]
		terminal_batch  = self.terminal_mem[batch]
		action_batch    = self.action_mem[batch]

		return state_batch, action_batch, reward_batch, new_state_batch, terminal_batch   

