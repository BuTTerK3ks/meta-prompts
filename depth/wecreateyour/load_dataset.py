import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
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
        try:
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

            image_path_extracted = os.path.join(self.data_path, 'image_extracted', base_filename + '.png')
            mask_path = os.path.join(self.data_path, 'mask_numpy', base_filename + '.npy')
            depth_path = os.path.join(self.data_path, 'depth_numpy', base_filename + '.npy')
            region_path = os.path.join(self.data_path, 'region', base_filename + '_region.pkl')

            # Load image, mask, and depth
            with Image.open(image_path) as img:
                image = np.array(img)
            mask = np.load(mask_path)
            depth = np.load(depth_path)
            region = np.load(region_path)
            with Image.open(image_path_extracted) as img:
                image_extracted = np.array(img)

            # Create a mask initially set to 1
            mask = np.ones_like(depth)

            # Set mask to 0 where depth is smaller than 10
            mask[depth < 10] = 0

            # Assuming image, mask, and image_extracted are numpy arrays (images)
            # Ensure that all images have the same number of channels (convert to RGB if needed)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            image_extracted_rgb = cv2.cvtColor(image_extracted, cv2.COLOR_BGR2RGB)

            # Concatenate images horizontally
            concatenated_image = cv2.hconcat([image_rgb, mask_rgb, image_extracted_rgb])

            # Display the concatenated image
            cv2.imshow(f'Images: {base_filename}', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #TODO Maske zuschneiden

            # Convert mask to have the same number of channels as the image
            # This expands the mask from (x, y) to (x, y, 3) by repeating the mask across the third dimension
            mask = np.stack([mask] * 3, axis=-1)

            # Apply the mask to the image and depth
            image = image * mask
            mask = mask[:, :, 0]
            depth = depth * mask  # Use the first channel of the expanded mask for depth

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
                mask = resize_keep_aspect(mask, self.resize_size, fill_value=0)  # Change fill_value if needed
                depth = resize_keep_aspect(depth, self.resize_size)

            # Normalize depth to range 0-10
            depth = depth / depth.max() * 10
            # Convert numpy arrays to PyTorch tensors
            image_tensor = torch.from_numpy(image).float() / 255.0  # Normalize image
            mask_tensor = torch.from_numpy(mask).long()  # Masks are typically long type
            depth_tensor = torch.from_numpy(depth).float()

            # Permute tensors to match PyTorch's NCHW format
            image_tensor = image_tensor.permute(2, 0, 1)
            #depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension to depth

            return {'image': image_tensor, 'mask': mask_tensor, 'depth': depth_tensor, 'filename': base_filename}

        except:

            print("Error loading file: " + str(base_filename))
            return self.__getitem__((idx + 1) % len(self))

if __name__ == "__main__":
    dataset_path = '/home/grannemann/Allgemein/Christian/LOOXIS/wecreateyour'
    id = ["3067173"]


    test_loader = ThreeDCDataset(dataset_path, id)

    for batch_idx, batch in test_loader:
        image = batch['image']
        mask = batch['mask']
        depth = batch['depth']

        print("batch")





    print("Ende")