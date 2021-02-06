from WorldModel import Enve 
from ComputeEngine import DQN_Actor, DoubleDQN_Actor

import torch
import numpy as np
import matplotlib.pyplot as plt

def train_QNetwork(n_games):	
	game_scores, game_losses = [], []

	for i in range(n_games):
		observation, reward, done = env.reset()

		score, counter = 0, 0

		while not env.done:
			action = agent.choose_action(observation)
			observation_, reward, done = env.step(action)
			score += reward

			agent.store_transition(observation, action, reward, observation_, done)
			agent.learn()
			observation = observation_
			counter+=1

		game_scores.append(score)
		avg_score = np.mean(game_scores[-100:], dtype=float)

		if i>100:
			if (i+1) % 10 == 0:
				agent.writer.add_scalar("Avg. Score", avg_score, i+1)
				print("Game: {} AvgScore: {:.3f}".format(i+1, avg_score))
			agent.writer.add_scalar("Score", score, i)

		if (i+1) % 4000 == 0:
			agent.save_ckpt(i+1, "{:.3f}".format(avg_score))
	
	agent.writer.close()

def eval_QNetwork(model_path, startIndex=100, endIndex=130, soc=0.6):

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
	# agent = DoubleDQN_Actor(gamma=0.99, epsilon=1, lr=0.001, input_dims=4, batch_size=32, num_actions=3)

	# train_QNetwork(20_000)

	agent = DQN_Actor(gamma=0.99, epsilon=1, lr=0.001, input_dims=4, batch_size=32, num_actions=3)

	train_QNetwork(20_000)
	print("...done...")

	# python3 -m tensorboard.main --logdir=~/my/training/dir
	
		
		






