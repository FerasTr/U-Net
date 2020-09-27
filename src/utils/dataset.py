import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, data_folder):
        self.image_dir = os.path.join(data_folder, "input")
        self.label_dir = os.path.join(data_folder, "target")
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
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # # Random horizontal flipping
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # image = TF.rotate(image, 90)
        # mask = TF.rotate(mask, 90)
        # image = TF.rotate(image, 180)
        # mask = TF.rotate(mask, 180)
        # image = TF.rotate(image, 270)
        # mask = TF.rotate(mask, 270)

        # Transform to tensor
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

        img, mask = self.transform(img, mask)
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
