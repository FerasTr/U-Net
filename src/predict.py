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

from utils.vis import *
from utils.metrics import *

import utils.params

wandb_track = False
if wandb_track:
    import wandb

    wandb.init(project="unet-med")

data_folder = "../data/"
model_path = "../model/"
model_name = "RMSprop_50e_0001_0047619_model"
save_path = model_path + model_name + "/"


def predict(net, n_channles, n_classes):

    dataset_test = Loader(data_folder, mode="test")
    test_loader = DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    n_test = int(len(dataset_test))
    net.eval()
    save = True
    show = False

    tavg_acc = 0
    tavg_class_acc = [0] * n_classes
    avg_inter_over_uni = [0] * n_classes
    conf_matrix = torch.zeros(n_classes, n_classes)

    for idx, i in enumerate(test_loader):
        print(idx)

        with torch.no_grad():
            input = i["image"]
            masks = i["mask"]

            input = input.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)

            output = net(input)

            conf_matrix = confusion_matrix(output, masks, conf_matrix)
            avg_inter_over_uni += iou(output, masks, n_classes)
            tavg_acc += multi_acc(output, masks)
            tavg_class_acc = np.add(
                np.array(tavg_class_acc),
                np.array(multi_acc_class(output, masks, n_classes)),
            )

            input = input.cpu().squeeze()
            input = transforms.ToPILImage()(input)

            masks = masks.cpu().squeeze()
            if n_classes == 3:
                masks = mask_to_image_3(masks)
            elif n_classes == 2:
                masks = mask_to_image_2(masks)

            binary_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            if n_classes == 3:

                fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(
                    1, 5, figsize=(20, 10), sharey=True
                )

            elif n_classes == 2:

                fig, (ax0, ax1, ax2, ax4) = plt.subplots(
                    1, 4, figsize=(20, 10), sharey=True
                )

            ax0.set_title("Input Image")
            ax1.set_title("Background Class")
            ax2.set_title("Dendrite Class")
            if n_classes == 3:
                ax3.set_title("Cell Class")
            ax4.set_title("Predicted Mask")

            ax0.imshow(input)
            ax1.imshow(binary_mask == 0)

            if n_classes == 3:
                ax2.imshow(binary_mask == 2)
                ax3.imshow(binary_mask == 1)
            elif n_classes == 2:
                ax2.imshow(binary_mask == 1)

            if n_classes == 3:
                img = mask_to_image_3(binary_mask)
            elif n_classes == 2:
                img = mask_to_image_2(binary_mask)

            ax4.imshow(img)
            if save:
                fig.savefig(save_path + str(idx + 1) + "_classes.png")
            if show:
                plt.show()
            plt.close(fig)

            if save:
                img.save(save_path + str(idx + 1) + "_out.png")
            if show:
                fig1, x = plt.subplots(nrows=1, ncols=1, sharey=True)
                x.imshow(img)
                plt.show()
                plt.close(fig1)

            fig2, (x1, x2, x3) = plt.subplots(nrows=1, ncols=3, sharey=True)
            x1.imshow(input)
            x2.imshow(masks)
            x3.imshow(img)
            if save:
                fig2.savefig(save_path + str(idx + 1) + "_img_truth_out.png")
            if show:
                plt.show()
            plt.close(fig2)

    tavg_acc /= n_test
    tavg_class_acc /= n_test
    avg_inter_over_uni /= n_test

    if wandb_track:
        wandb.log({"Test Average Accuracy": tavg_acc})
        if n_classes == 3:
            wandb.log(
                {
                    "Test Average Background Accuracy": tavg_class_acc[0],
                    "Test Average Cells Accuracy": tavg_class_acc[1],
                    "Test Average Dendrites Accuracy": tavg_class_acc[2],
                }
            )

            wandb.log(
                {
                    "Test Average IoU for Background": avg_inter_over_uni[0],
                    "Test Average IoU for Neurons": avg_inter_over_uni[1],
                    "Test Average IoU for Dendrites": avg_inter_over_uni[2],
                }
            )
        elif n_classes == 2:
            wandb.log(
                {
                    "Test Average Background Accuracy": tavg_class_acc[0],
                    "Test Average Dendrites Accuracy": tavg_class_acc[1],
                }
            )

            wandb.log(
                {
                    "Test Average IoU for Background": avg_inter_over_uni[0],
                    "Test Average IoU for Dendrites": avg_inter_over_uni[1],
                }
            )

    for i in range(n_classes):
        r, s, p, f = confusion_matrix_metrics(i, n_classes, conf_matrix)

        msg_r = "Test Class {} Recall".format(i)
        msg_s = "Test Class {} Specificity".format(i)
        msg_p = "Test Class {} Precision".format(i)
        msg_f = "Test Class {} F1-score".format(i)

        print(msg_r + " " + str(r))
        print(msg_s + " " + str(s))
        print(msg_p + " " + str(p))
        print(msg_f + " " + str(f))

        if wandb_track:
            wandb.log(
                {msg_r: r, msg_s: s, msg_p: p, msg_f: f,}
            )


if __name__ == "__main__":
    n_classes = utils.params.n_classes
    n_channels = utils.params.n_channels
    net = UNet(n_channels=n_channels, n_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device=device)
    try:
        os.mkdir(save_path)
    except:
        pass
    net.load_state_dict(
        torch.load(model_path + model_name + ".pth", map_location=device)
    )
    predict(net, n_channels, n_classes=n_classes)
