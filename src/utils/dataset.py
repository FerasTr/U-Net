import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, data_folder, test=False):
        self.image_dir = os.path.join(data_folder, "input")
        self.label_dir = os.path.join(data_folder, "target")
        self.test = test
        if self.test:
            self.image_dir = os.path.join(data_folder, "test")
            self.label_dir = os.path.join(data_folder, "test-target") 
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
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        rotation = [0, 90, 180, 270]
        if random.random() > 0.5:
            r = random.choice(rotation)
            image = TF.rotate(image,r)
            mask = TF.rotate(mask,r)

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

        if not self.test:
            img, mask = self.transform(img, mask)
        img, mask = self.transform_to_tensor(img, mask)
        if not self.test:
           mask = self.mask_to_class(mask)
        return {'image': img, 'mask': mask}


if __name__ == "__main__":
    data_folder = "../../data/"
    demo = Loader(data_folder)
    data = demo.__getitem__(5)
    show = True
    if show:
        print(data["image"])
        print(data["mask"])
