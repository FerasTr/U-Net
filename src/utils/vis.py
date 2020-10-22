import torch
import numpy as np
from PIL import Image

def mask_to_image(mask):
    mask = torch.tensor(mask)

    black = mask == 0
    red = mask == 1
    white = mask == 2

    image = torch.stack([red, black, white],dim=2).int().numpy() * 255
    return Image.fromarray((image).astype(np.uint8))

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
