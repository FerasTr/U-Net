import numpy as np
from PIL import Image
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataset import Loader
from model.unet import UNet

data_folder = "../data/"
model_path = "../model/model.pth"


def predict(net):

    testset = Loader(data_folder)
    test_loader = DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    net.eval()
    for i in test_loader:
        with torch.no_grad():
            input = i["image"]

            input = input.to(device=device, dtype=torch.float32)

            output = net(input)

            probs = F.softmax(output, dim=1)
            probs = probs.squeeze(0)

            full_mask = probs.squeeze().cpu().numpy()


            _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            ax1.imshow(full_mask[0,:,:].squeeze())
            ax2.imshow(full_mask[1,:,:].squeeze())
            ax3.imshow(full_mask[2,:,:].squeeze())
            plt.show()

            full_mask = np.argmax(full_mask, 0)

            img = mask_to_image(full_mask)

            plt.imshow(img)
            plt.show()
            # input_array = input.squeeze().detach().numpy()
            # output_array = output.argmax(2) * 255
            # input_img = Image.fromarray(input_array)
            # output_img = Image.fromarray(output_array.astype(dtype=np.uint16)).convert("L")
            # input_img.show()
            # output_img.show()
    return full_mask

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB')

if __name__ == "__main__":
    net = UNet(n_channels=3, n_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    net.load_state_dict(torch.load(model_path, map_location=device))
    predict(net)