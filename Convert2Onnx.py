import torch

from Networks import DQN

def main():
	model = DQN(lr=0.001, input_dims=4, fc1_dim=256, fc2_dim=256, n_actions=3, checkpoint_path=".", name="DQN")
	model.load_state_dict(torch.load("Trained_models/Strategy1/ShortSequence/DDQN_Model 5.47.31 PM/Q_Local_20000_Weights_score_21.569.pt"))
	model.eval()
	dummy_input = torch.zeros(1,4)
	torch.onnx.export(model,dummy_input, "DDQN_3.onnx", verbose=True )
if __name__ == '__main__':
		main();
		