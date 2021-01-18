from WorldModel import Enve 
from Actor import Actor
import numpy as np

if __name__ == "__main__":
    env = Enve()

    agent = Actor(DataFile_path="Data/PriceData.csv", gamma=0.99, epsilon=1, lr=0.001, 
                input_dims=4, batch_size=64, n_actions=3)
    scores, eps_history = [],[]

    n_games = 10000
    for i in range(n_games):
        score = 0
        observation, reward, done = env.reset()
        counter = 0

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
        
        






