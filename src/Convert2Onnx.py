import torch

from src.Networks import DQN

def main():
	model = DQN(lr=0.001, input_dims=4, fc1_dim=256, fc2_dim=256, n_actions=3, checkpoint_path=".", name="DQN")
	model.load_state_dict(torch.load("Trained_models/Strategy3/episode_100_200/3/DQN/DQN_6000_Weights_score_54.326.pt"))
	model.eval()
	dummy_input = torch.zeros(1,4)
	torch.onnx.export(model,dummy_input, "DQN_Long_S3_3.onnx", verbose=True )
if __name__ == '__main__':
		main();
		