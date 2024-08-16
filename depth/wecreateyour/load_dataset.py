import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt


class ThreeDCDataset(Dataset):
    def __init__(self, data_path, ids, resize_size=(448, 576), is_train=True):
        """
        Args:
            data_path (string): Path to the directory with all the data.
            split_file (string): Path to the text file containing IDs for validation/test split.
            resize_size (tuple, optional): Desired output size. Default is (448, 576).
            is_train (bool): Flag to indicate if the dataset is used for training. Default is True.
        """
        self.data_path = data_path
        self.resize_size = resize_size
        self.is_train = is_train

        # List of filenames for images, masks, and depth
        #self.image_filenames = [f for f in os.listdir(os.path.join(data_path, 'image_numpy')) if f.endswith('.npy')]
        #self.mask_filenames = [f for f in os.listdir(os.path.join(data_path, 'mask_numpy')) if f.endswith('.npy')]
        #self.depth_filenames = [f for f in os.listdir(os.path.join(data_path, 'depth_numpy')) if f.endswith('.npy')]

        # Filter filenames based on split
        self.ids = ids

        # Extensions for image
        self.possible_extensions = ['.png', '.jpg', '.jpeg']

        print(f"Dataset initialized. {'Training' if is_train else 'Validation/Test'} mode. Total samples: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base_filename = self.ids[idx]
    #try:
        # Find the image file with the correct extension
        image_path = None
        for ext in self.possible_extensions:
            potential_path = os.path.join(self.data_path, 'images', base_filename + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            raise FileNotFoundError(
                f"No image file found for {base_filename} with extensions {self.possible_extensions}")

        depth_path = os.path.join(self.data_path, 'depth_numpy', base_filename + '.npy')
        region_path = os.path.join(self.data_path, 'region', base_filename + '_region.pkl')

        # Load image and depth
        with Image.open(image_path) as img:
            image = np.array(img)
        depth = np.load(depth_path)

        # Load region (using pickle)
        with open(region_path, 'rb') as region_file:
            region = pickle.load(region_file)

        # Create a mask with the same size as the image, initialized to 0
        mask_in_image_dim = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Set mask to 1 where depth is greater than or equal to 10
        mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)
        mask[depth >= 10] = 1

        # Extract region information from the pickle file
        top_left_x, top_left_y = region['top_left']
        region_height = region['height']
        region_width = region['width']

        # Scale the mask to fit the region dimensions
        scaled_mask = cv2.resize(mask, (region_width, region_height), interpolation=cv2.INTER_NEAREST)

        # Incorporate the cropped mask into the mask_in_image_dim using the region
        mask_in_image_dim[top_left_y:top_left_y + region_height, top_left_x:top_left_x + region_width] = scaled_mask



        if self.resize_size:
            # Resize while keeping aspect ratio
            def resize_keep_aspect(image, target_size, fill_value=0):
                ih, iw = image.shape[:2]
                th, tw = target_size
                scale = min(tw / iw, th / ih)

                nw = int(iw * scale)
                nh = int(ih * scale)

                image_resized = cv2.resize(image, (nw, nh))

                if len(image.shape) == 3:  # For RGB images
                    new_image = np.full((th, tw, 3), fill_value, dtype=image.dtype)
                else:  # For masks and depth maps
                    new_image = np.full((th, tw), fill_value, dtype=image.dtype)

                new_image[(th - nh) // 2:(th - nh) // 2 + nh, (tw - nw) // 2:(tw - nw) // 2 + nw] = image_resized
                return new_image

            image = resize_keep_aspect(image, self.resize_size)
            mask_in_image_dim = resize_keep_aspect(mask_in_image_dim, self.resize_size, fill_value=0)

        '''

        # Convert the final mask to have the same number of channels as the image
        mask_rgb = np.stack([mask_in_image_dim] * 3, axis=-1)

        # Apply the mask to the image
        masked_image = image * mask_rgb

        # Ensure all images are of the same data type
        mask_rgb = mask_rgb * 255
        mask_rgb = mask_rgb.astype(image.dtype)
        masked_image = masked_image.astype(image.dtype)

        # Convert to RGB if needed (depends on original image mode)
        if image.shape[2] == 1:  # If grayscale, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Visualization: Concatenate and display the images
        concatenated_image = cv2.hconcat([image_rgb, mask_rgb, masked_image])
        cv2.imshow(f'Images: {base_filename}', concatenated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''


        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image).float() / 255.0  # Normalize image
        mask_tensor = torch.from_numpy(mask_in_image_dim).long()  # Masks are typically long type

        # Permute tensors to match PyTorch's NCHW format
        image_tensor = image_tensor.permute(2, 0, 1)

        return {'image': image_tensor, 'mask': mask_tensor, 'filename': base_filename}

        '''
        except Exception as e:
            print(f"Error processing {base_filename}: {e}")
            raise
        '''








if __name__ == "__main__":
    dataset_path = '/home/grannemann/Allgemein/Christian/LOOXIS/wecreateyour'
    id = ["3067173", "3067174"]


    test_loader = ThreeDCDataset(dataset_path, id)

    for batch_idx, batch in test_loader:
        image = batch['image']
        mask = batch['mask']
        depth = batch['depth']

        print("batch")





    print("Ende")