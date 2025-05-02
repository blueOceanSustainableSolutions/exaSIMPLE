import torch

# Path to the checkpoint file
checkpoint_path = "<checkpoint_path>" # e.g. "checkpoints/epoch=58-step=156762.ckpt"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# Print the top-level keys
print("Checkpoint Keys:")
print(checkpoint.keys())

# Print details about state_dict
if "state_dict" in checkpoint:
    print("\nModel State Dict Keys:")
    print(checkpoint["state_dict"].keys())

# Check for epoch
if "epoch" in checkpoint:
    print(f"\nCheckpoint saved at epoch: {checkpoint['epoch']}")

# Check for optimizer state
if "optimizer_states" in checkpoint:
    print("\nOptimizer state found in checkpoint.")

# Print hyperparameters if they exist
if "hyper_parameters" in checkpoint:
    print("\nHyperparameters:")
    for key, value in checkpoint["hyper_parameters"].items():
        print(f"  {key}: {value}")

# Print other metadata
for key in checkpoint.keys():
    if key not in ["state_dict", "optimizer_states", "epoch", "hyper_parameters"]:
        print(f"\nAdditional key '{key}':")
        print(checkpoint[key])
