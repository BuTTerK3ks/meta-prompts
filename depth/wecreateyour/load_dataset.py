import os
import struct
import cv2
import lzma
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from matplotlib import pyplot as plt
import json


class ThreeDCDataset(Dataset):
    def __init__(self, data_path, crop_size=(448, 576), scale_size=None):
        """
        Args:
            data_path (string): Path to the directory with all the data.
            crop_size (tuple, optional): Desired output size. Default is (448, 576).
            scale_size (tuple, optional): Size to which images should be scaled before cropping. Default is None.
        """
        if crop_size[0] > 479:
            scale_size = (int(crop_size[0] * 640 / 480), crop_size[0])
        self.scale_size = scale_size
        self.is_train = True

        self.data_path = os.path.join(data_path, '3dc_data')

        # Check if the data_path is actually a directory
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError(f"{self.data_path} is not a directory.")

        # Scan the directory and generate the list of '.3dc' file names
        self.filenames_list = [f for f in os.listdir(self.data_path) if f.endswith('.3dc')]

        print(f"Dataset: 3DC Data")
        print(f"# of images: {len(self.filenames_list)}")

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.filenames_list[idx].split(' ')[0])
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        class_id = 0

        # Use a modified read_3dc_file method to read the .3dc file and obtain the RGB image
        depth, image_rgba = self.read_3dc_file(file_path=img_path)

        # Convert RGBA to RGB by discarding the Alpha channel
        image = image_rgba[:, :, :3].astype(np.uint8)



        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()  # Ensure the tensor is of type float
        image_tensor = image_tensor / 255.0
        depth_tensor = torch.from_numpy(depth).float()

        # Permute the image tensor from HxWxC to CxHxW
        # Assuming the original image is in HxWxC format
        image_tensor = image_tensor.permute(2, 0, 1)

        return {'image': image_tensor, 'depth': depth_tensor, 'filename': filename, 'class_id': class_id}

    def convert_10bit_rgba_to_8bit_rgb(self, image):
        # Assuming 'image' is a numpy array with shape (height, width, 4) and dtype=np.uint16
        # Normalize from 10-bit to 8-bit (0-255)
        image_8bit = (image / 1023.0) * 255.0
        image_8bit = image_8bit.astype(np.uint8)

        # Keep only the first three channels (RGB)
        rgb_image = image_8bit[:, :, :3]

        return rgb_image

    @staticmethod
    def argb_to_rgba_vectorized(argb_values):
        """Convert a one-dimensional numpy array of ARGB values to RGBA values."""
        argb_values = np.asarray(argb_values, dtype=np.uint32)
        a = (argb_values >> 24) & 0xFF
        b = (argb_values >> 16) & 0xFF
        g = (argb_values >> 8) & 0xFF
        r = argb_values & 0xFF
        rgba_values = np.stack((r, g, b, a), axis=-1)
        return rgba_values

    @staticmethod
    def read_3dc_file(file_path):
        """Reads .3dc file and returns numpy arrays for z values and RGBA."""
        decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
        with open(file_path, 'rb') as file:
            identifier, = struct.unpack('H', file.read(2))
            if identifier != 18739:
                raise ValueError(f'Invalid file identifier: {identifier}. Expected: 18739')

            version, header_size = struct.unpack('2H', file.read(4))
            if version != 19 or header_size != 64:
                raise ValueError(f'Unexpected 3DC version ({version}) or header size ({header_size}).')

            file.seek(24)
            num_points, width, height, pack_size = struct.unpack('4I', file.read(16))
            if not all([num_points, width, height, pack_size]):
                raise ValueError('Failed to read file metadata properly.')

            height += 3
            width += 3

            file.seek(header_size)
            compressed_data = file.read(pack_size)
            properties = compressed_data[:5]
            uncompressed_size = width * height * 8
            lzma_header = properties + struct.pack('<Q', uncompressed_size)
            custom_compressed_data = lzma_header + compressed_data[5:]

            decompressed_data = decompressor.decompress(custom_compressed_data)
            expected_size = width * height * 8
            if len(decompressed_data) != expected_size:
                raise ValueError('Decompressed data size does not match the expected size.')

            height -= 2
            width -= 2

            dt = np.dtype([('z', np.float32), ('rgb', np.uint32)])
            points = np.frombuffer(decompressed_data, dtype=dt)[:width * height].reshape((width, height))
            points = np.rot90(points)

            z_values = points['z']
            rgba_values = ThreeDCDataset.argb_to_rgba_vectorized(points['rgb'].flatten()).reshape((height, width, 4))

            return z_values, rgba_values