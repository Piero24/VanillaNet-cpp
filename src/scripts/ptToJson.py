# MACOS
# pip3 install torch torchvision torchaudio
import torch
import json

# Load the PyTorch model
pt_path = './Resources/output/weights/mnist_fc128_relu_fc10_log_softmax'

# IF CUDA
# state_dict = torch.load(pt_path + '.pt')

# Load the state dictionary, mapping it to CPU
state_dict = torch.load(pt_path + '.pt', map_location=torch.device('cpu'))

# Convert the state_dict (OrderedDict) to a regular dictionary for easier processing
weights_biases = {name: param.numpy().tolist() for name, param in state_dict.items()}

# Save weights and biases to a JSON file
with open(pt_path + '_weights_biases.json', 'w') as f:
    json.dump(weights_biases, f, indent=4)

print(f"Weights and biases successfully saved to {pt_path}_weights_biases.json")