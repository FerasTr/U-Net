import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset


class Loader(Dataset):
    def __init__(self, data_folder, mode):
        self.mode = mode
        if mode == "train":
            self.image_dir = os.path.join(data_folder, "input")
            self.label_dir = os.path.join(data_folder, "target")
        if mode == "test":
            self.image_dir = os.path.join(data_folder, "test_input")
            self.label_dir = os.path.join(data_folder, "test_target")
        if mode == "val":
            self.image_dir = os.path.join(data_folder, "val_input")
            self.label_dir = os.path.join(data_folder, "val_target")
        self.images = [
            file for file in os.listdir(self.image_dir) if not file.startswith(".")
        ]
        self.labels = [
            file for file in os.listdir(self.label_dir) if not file.startswith(".")
        ]

    def mask_to_class(self, mask):
        target = mask
        h, w = target.shape[0], target.shape[1]
        masks = torch.empty(h, w, dtype=torch.long)
        colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
        target = target.permute(2, 0, 1).contiguous()
        mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
        for k in mapping:
            idx = target == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2)
            validx = idx.sum(0) == 3
            masks[validx] = torch.tensor(mapping[k], dtype=torch.long)
        return masks

    def transform(self, image, mask):
        rotation = [0, 90, 180, 270]
        if random.random() > 0.5:
            r = random.choice(rotation)
            image = TF.rotate(image, r)
            mask = TF.rotate(mask, r)
        pix_add = random.randint(-40, 40)
        image = self.change_brightness(image, pix_add)
        return image, mask

    def transform_to_tensor(self, image, mask):
        image = np.array(image) / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(np.array(image)).float()
        mask = torch.from_numpy(np.array(mask))
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = os.path.join(self.image_dir, self.images[index])
        mask_file = os.path.join(self.label_dir, self.labels[index])

        img = Image.open(img_file)
        mask = Image.open(mask_file)

        if self.mode == "train":
            img, mask = self.transform(img, mask)
        img, mask = self.transform_to_tensor(img, mask)
        mask = self.mask_to_class(mask)
        return {"image": img, "mask": mask}


    def change_brightness(self,image, value):
        """
        Args:
            image : numpy array of image
            value : brightness
        Return :
            image : numpy array of image with brightness added
        """
        image = np.array(image)
        image = image.astype("int16")
        image = image + value
        image = self.ceil_floor_image(image)
        return image


    def ceil_floor_image(self,image):
        """
        Args:
            image : numpy array of image in datatype int16
        Return :
            image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
        """
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype("uint8")
        return image


if __name__ == "__main__":
    data_folder = "/home/fdxd/Workspace/U-Net/data"
    demo = Loader(data_folder, mode="train")
    data = demo.__getitem__(0)
    show = True
    if show:
        print(data["image"].shape)
        print(data["mask"].shape)
