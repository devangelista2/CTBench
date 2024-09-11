import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from miscellanous import utilities


class ImageDataset(Dataset):
    """
    Implements a Dataset subclass that reads .png images from a folder of data
    (where each element is assumed to be a .png representing gray-scale image)
    and converts it to either a numpy array or a pytorch Tensor.

    Arguments:
        data_path: str, (Relative) path to the dataset.
        numpy: bool, if True, returns a numpy array, a pytorch Tensor is returned
                    otherwise. Default: False.
    """

    def __init__(self, data_path, transforms=None):
        self.data_path = data_path

        self.img_name_list = sorted(
            glob.glob(os.path.join(self.data_path, "*", "*.png"))
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        img = Image.open(self.img_name_list[index])
        img = utilities.normalize(np.expand_dims(np.array(img)[:, :, 0], 0))

        # Convert to pytorch tensor
        img = torch.Tensor(img)

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def get_name(self, index):
        return self.img_name_list[index].split("\\")[-1]
