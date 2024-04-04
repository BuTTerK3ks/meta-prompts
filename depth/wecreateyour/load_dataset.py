import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class ThreeDCDataset(Dataset):
    def __init__(self, data_path, ids, crop_size=(448, 576), is_train=True):
        """
        Args:
            data_path (string): Path to the directory with all the data.
            split_file (string): Path to the text file containing IDs for validation/test split.
            crop_size (tuple, optional): Desired output size. Default is (448, 576).
            is_train (bool): Flag to indicate if the dataset is used for training. Default is True.
        """
        self.data_path = data_path
        self.crop_size = crop_size
        self.is_train = is_train

        # List of filenames for images, masks, and depth
        self.image_filenames = [f for f in os.listdir(os.path.join(data_path, 'image_numpy')) if f.endswith('.npy')]
        self.mask_filenames = [f for f in os.listdir(os.path.join(data_path, 'mask_numpy')) if f.endswith('.npy')]
        self.depth_filenames = [f for f in os.listdir(os.path.join(data_path, 'depth_numpy')) if f.endswith('.npy')]

        # Filter filenames based on split
        self.ids = ids

        print(f"Dataset initialized. {'Training' if is_train else 'Validation/Test'} mode. Total samples: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base_filename = self.ids[idx]
        image_path = os.path.join(self.data_path, 'image_numpy', base_filename + '.npy')
        mask_path = os.path.join(self.data_path, 'mask_numpy', base_filename + '.npy')
        depth_path = os.path.join(self.data_path, 'depth_numpy', base_filename + '.npy')

        # Load image, mask, and depth
        image = np.load(image_path)
        mask = np.load(mask_path)
        depth = np.load(depth_path)

        # Assuming the images are already in RGB format
        if self.crop_size:
            image = cv2.resize(image, self.crop_size)
            mask = cv2.resize(mask, self.crop_size, interpolation=cv2.INTER_NEAREST)  # Use nearest interpolation for masks
            depth = cv2.resize(depth, self.crop_size)

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image).float() / 255.0  # Normalize image
        mask_tensor = torch.from_numpy(mask).long()  # Masks are typically long type
        depth_tensor = torch.from_numpy(depth).float() / 100

        # Permute tensors to match PyTorch's NCHW format
        image_tensor = image_tensor.permute(2, 0, 1)
        #depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension to depth

        return {'image': image_tensor, 'mask': mask_tensor, 'depth': depth_tensor, 'filename': base_filename}
