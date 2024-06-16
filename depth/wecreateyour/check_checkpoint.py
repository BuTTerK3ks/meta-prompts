import torch

def inspect_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU for inspection

    # Inspect the model state dictionary
    if 'model' in checkpoint:
        print("\nLayers and parameters in model state dict:")
        model_state_dict = checkpoint['model']
        for layer_name, tensor in model_state_dict.items():
            print(f"{layer_name}: {tensor.size()}")

    # Inspect the optimizer state dictionary
    if 'optimizer' in checkpoint:
        print("\nOptimizer state:")
        optimizer_state_dict = checkpoint['optimizer']
        for key in optimizer_state_dict:
            print(key)
            # Further inspection of optimizer's state, if needed
            if isinstance(optimizer_state_dict[key], dict):  # Checking if the entry is a dictionary
                for subkey in optimizer_state_dict[key]:
                    print(f"  {subkey}: {type(optimizer_state_dict[key][subkey])}")

# Usage
inspect_checkpoint('/home/grannemann/PycharmProjects/meta-prompts/depth/logs/wecreateyour_body_Stable_Pre_epoch_01_/last.ckpt')