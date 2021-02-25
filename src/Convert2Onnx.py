import torch

from Networks import DQN

def main():
	model = DQN(lr=0.001, input_dims=4, fc1_dim=256, fc2_dim=256, n_actions=3, checkpoint_path=".", name="DQN")
	model.load_state_dict(torch.load("./Trained_models/Strategy1/episode_100_200/DQN/DQN_10000_Weights_score_74.848.pt"))
	model.eval()
	dummy_input = torch.zeros(1,4)
	torch.onnx.export(model,dummy_input, "DQN_Long_S1.onnx", verbose=True )
if __name__ == '__main__':
		main();
		