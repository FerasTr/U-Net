import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms

from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataset import Loader
from model.unet import UNet

data_folder = "../data/"
model_path = "../model/"
model_name = "RMSprop_100e_0001_03051_model"
save_path = model_path + model_name + '/'


def predict(net):

    testset = Loader(data_folder, test=True)
    test_loader = DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    net.eval()
    print(len(testset))
    for idx, i in enumerate(test_loader):
        with torch.no_grad():
            input = i["image"]
            input = input.to(device=device, dtype=torch.float32)
            output = net(input)
            input = input.cpu().squeeze()
            input = transforms.ToPILImage()(input)
            probs = F.softmax(output, dim=1)
            probs = probs.squeeze(0)

            full_mask = probs.squeeze().cpu().numpy()

            fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(20,10), sharey=True)
            ax0.set_title('Input Image')
            ax0.imshow(input)
            ax1.set_title('Background Class')
            ax1.imshow(full_mask[0, :, :].squeeze())
            ax2.set_title('Neuron Class')
            ax2.imshow(full_mask[1, :, :].squeeze())
            ax3.set_title('Dendrite Class')
            ax3.imshow(full_mask[2, :, :].squeeze())

            full_mask = np.argmax(full_mask, 0)

            img = mask_to_image(full_mask)

            ax4.set_title('Predicted Mask')
            ax4.imshow(img)

            imm = get_concat_h(img, input)
            imm.save(save_path + str(idx + 1) + '_full.png')
            fig.savefig(save_path + str(idx + 1) + '.png')


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == "__main__":
    net = UNet(n_channels=3, n_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device=device)
    try:
        os.mkdir(save_path)
    except:
        pass
    net.load_state_dict(torch.load(model_path + model_name + '.pth', map_location=device))
    predict(net)
