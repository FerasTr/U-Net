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
    save = True
    show = False
    for idx, i in enumerate(test_loader):
        print(idx)
        with torch.no_grad():
            input = i["image"]
            input = input.to(device=device, dtype=torch.float32)

            output = net(input)

            input = input.cpu().squeeze()
            input = transforms.ToPILImage()(input)

            binary_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(20,10), sharey=True)

            ax0.set_title('Input Image')
            ax1.set_title('Background Class')
            ax2.set_title('Neuron Class')
            ax3.set_title('Dendrite Class')
            ax4.set_title('Predicted Mask')

            ax0.imshow(input)
            ax1.imshow(binary_mask==0)
            ax2.imshow(binary_mask==1)
            ax3.imshow(binary_mask==2)

            img = mask_to_image(binary_mask)

            ax4.imshow(img)
            if save:
                fig.savefig(save_path + str(idx + 1) + '_classes.png')
            if show:
                plt.show()
            plt.close(fig)

            fig1, x = plt.subplots(nrows=1, ncols=1, sharey=True)
            x.imshow(img)
            if save:
                fig1.savefig(save_path + str(idx + 1) + '_mask.png')
            if show:
                plt.show()
            plt.close(fig1)

            fig2, (x1,x2) = plt.subplots(nrows=1, ncols=2, sharey=True)
            x2.imshow(input)
            x1.imshow(img)
            if save:
                fig2.savefig(save_path + str(idx + 1) + '_img_mask.png')
            if show:
                plt.show()
            plt.close(fig2)


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
