from WorldModel import Enve 
from Actor import Actor

import torch
import numpy as np
import matplotlib.pyplot as plt

def train(n_games):	
	scores, eps_history = [],[]

	for i in range(n_games):
		score, counter = 0, 0
		observation, reward, done = env.reset()

		while not env.done:
			action = agent.choose_action(observation)
			observation_, reward, done = env.step(action)
			score += reward

			agent.store_transition(observation, action, reward, observation_, done)
			agent.learn()
			observation = observation_
			counter+=1
			
		scores.append(score)
		eps_history.append(agent.epsilon)
		avg_score = np.mean(scores[-100:])

		print(f"Game : {i} Score: {score}   AvgScore: {avg_score} ")

	print("__done__")
	torch.save(agent.Q_eval, "tensor.pt")

def eval(model_path, startIndex=100, endIndex=130, soc=0.6):

	observation, reward, done = env.test(startIndex, endIndex, soc)
	agent.Q_eval.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	if agent.Q_eval.device =='cpu':
		agent.Q_eval= torch.load(model_path, map_location=torch.device('cpu'))

	out = []
	out.append(observation)
	score=0
	while not done:
		action = agent.choose_action(observation, test=True)
		print(f"observation: {observation} : action: {action} ")
		observation_, reward, done = env.step(action)
		score += reward
		out.append(observation_)
		observation = observation_

	out = np.asarray(out)
	return out


if __name__ == "__main__":
	data_file = "Data/PriceData.csv"

	env 	= Enve(DataFile_path=data_file, max_charge=0.8, min_charge=0.2, rate=0.1, battery_cap=1500)
	agent = Actor(gamma=0.99, epsilon=1, lr=0.001, input_dims=4, batch_size=64, n_actions=3)

	# train(1000)
	# eval("tensor1.pt")
	
		
		






