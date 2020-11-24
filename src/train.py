import os
import random
import sys
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils.dataset import Loader
from model.unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

from utils.metrics import *
import utils.params

wandb_track = False

if wandb_track:
    import wandb

    wandb.init(project="unet-med")

data_folder = "../data/"
model_path = "../model/"
model_name = "RMSprop_50e_0001_0047619_model.pth"


def train_net(
    net,
    n_channels,
    n_classes,
    class_weights,
    epochs=50,
    batch_size=1,
    lr=0.0001,
    weight_decay=1e-8,
    momentum=0.99,
):
    print("Creating dataset for training...")
    dataset_train = Loader(data_folder , mode="train")
    dataset_val = Loader(data_folder , mode="val")
    n_val = int(len(dataset_val))
    n_train = int(len(dataset_train))

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    cw = torch.Tensor(class_weights).to(device=device)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device=device), reduction="mean")
    criterion = nn.CrossEntropyLoss(weight=cw, reduction="mean")

    if wandb_track:
        wandb.watch(net)

    tavg_lss = 0
    vavg_lss = 0
    tavg_acc = 0
    vavg_acc = 0
    tavg_class_acc = [0] * n_classes
    vavg_class_acc = [0] * n_classes
    avg_inter_over_uni = [0] * n_classes
    conf_matrix = torch.zeros(n_classes, n_classes)

    for epoch in range(epochs):

        net.train()

        tepoch_loss = 0
        tepoch_acc = 0
        vepoch_loss = 0
        vepoch_acc = 0
        tepoch_class_acc = [0] * n_classes
        vepoch_class_acc = [0] * n_classes
        inter_over_uni = [0] * n_classes

        for batch in train_loader:
            imgs = batch["image"]
            masks = batch["mask"]
            assert imgs.shape[1] == n_channels, (
                f"Network has been defined with {n_channels} input channels, "
                f"but loaded images have {imgs.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )
            imgs = imgs.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)

            masks_pred = net(imgs)

            loss = criterion(masks_pred, masks)
            # loss = loss.sum() / cw[masks].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch_loss += loss.item()
            tepoch_acc += multi_acc(masks_pred, masks)
            tepoch_class_acc = np.add(
                np.array(tepoch_class_acc),
                np.array(multi_acc_class(masks_pred, masks, n_classes)),
            )

        net.eval()
        for batch in val_loader:
            with torch.no_grad():
                imgs = batch["image"]
                masks = batch["mask"]
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, masks)
                # loss = loss.sum() / cw[masks].sum()

                conf_matrix = confusion_matrix(masks_pred, masks, conf_matrix)

                inter_over_uni += iou(masks_pred, masks, n_classes)
                vepoch_loss += loss.item()
                vepoch_acc += multi_acc(masks_pred, masks)
                vepoch_class_acc = np.add(
                    np.array(vepoch_class_acc),
                    np.array(multi_acc_class(masks_pred, masks, n_classes)),
                )
        tepoch_loss /= n_train
        tepoch_acc /= n_train
        tavg_lss += tepoch_loss
        tavg_acc += tepoch_acc
        tepoch_class_acc /= n_train
        tavg_class_acc = np.add(np.array(tavg_class_acc), np.array(tepoch_class_acc))

        vepoch_loss /= n_val
        vepoch_acc /= n_val
        vavg_lss += vepoch_loss
        vavg_acc += vepoch_acc
        vepoch_class_acc /= n_val
        vavg_class_acc = np.add(np.array(vavg_class_acc), np.array(vepoch_class_acc))

        inter_over_uni /= n_val
        avg_inter_over_uni += inter_over_uni

        print(
            "Epoch {0:}, Training loss: {1:.4f} [{2:.2f}%]  Validation loss: {3:.4f} [{4:.2f}%]".format(
                epoch + 1, tepoch_loss, tepoch_acc, vepoch_loss, vepoch_acc,
            )
        )
        if n_classes == 3:
            print(
                "Val Backg Accuracy: [{0:.2f}%] Val Cells Accuracy: [{1:.2f}%] Val Dendrites Accuracy: [{2:.2f}%]".format(
                    vepoch_class_acc[0], vepoch_class_acc[1], vepoch_class_acc[2],
                )
            )
        elif n_classes == 2:
            print(
                "Val Backg Accuracy: [{0:.2f}%] Val Dendrites Accuracy: [{1:.2f}%]".format(
                    vepoch_class_acc[0], vepoch_class_acc[1],
                )
            )
       
        if wandb_track:
            wandb.log({"Training Accuracy": tepoch_acc, "Training Loss": tepoch_loss})
            wandb.log(
                {"Validation Accuracy": vepoch_acc, "Validation Loss": vepoch_loss}
            )
            if n_classes == 3:
                wandb.log(
                    {
                        "Training Background Accuracy": tepoch_class_acc[0],
                        "Training Cells Accuracy": tepoch_class_acc[1],
                        "Training Dendrites Accuracy": tepoch_class_acc[2],
                    }
                )
                wandb.log(
                    {
                        "Validation Background Accuracy": vepoch_class_acc[0],
                        "Validation Cells Accuracy": vepoch_class_acc[1],
                        "Validation Dendrites Accuracy": vepoch_class_acc[2],
                    }
                )
                wandb.log(
                    {
                        "Validation IoU for Background": inter_over_uni[0],
                        "Validation IoU for Cells": inter_over_uni[1],
                        "Validation IoU for Dendrites": inter_over_uni[2],
                    }
                )
            elif n_classes == 2:
                wandb.log(
                    {
                        "Training Background Accuracy": tepoch_class_acc[0],
                        "Training Dendrites Accuracy": tepoch_class_acc[1],
                    }
                )
                wandb.log(
                    {
                        "Validation Background Accuracy": vepoch_class_acc[0],
                        "Validation Dendrites Accuracy": vepoch_class_acc[1],
                    }
                )
                wandb.log(
                    {
                        "Validation IoU for Background": inter_over_uni[0],
                        "Validation IoU for Dendrites": inter_over_uni[1],
                    }
                )

    tavg_acc /= epochs
    tavg_class_acc /= epochs
    tavg_lss /= epochs

    vavg_acc /= epochs
    vavg_class_acc /= epochs
    vavg_lss /= epochs

    avg_inter_over_uni /= epochs

    for i in range(n_classes):
        r, s, p, f = confusion_matrix_metrics(i, n_classes, conf_matrix)

        msg_r = "Val Class {} Recall".format(i)
        msg_s = "Val Class {} Specificity".format(i)
        msg_p = "Val Class {} Precision".format(i)
        msg_f = "Val Class {} F1-score".format(i)

        print(msg_r + " " + str(r))
        print(msg_s + " " + str(s))
        print(msg_p + " " + str(p))
        print(msg_f + " " + str(f))

        if wandb_track:
            wandb.log(
                {msg_r: r, msg_s: s, msg_p: p, msg_f: f,}
            )

    if wandb_track:
        wandb.log(
            {
                "Train Average Accuracy": tavg_acc,
                "Validation Average Accuracy": vavg_acc,
            }
        )
        wandb.log({"Train Average Loss": tavg_lss, "Validation Average Loss": vavg_lss})
        if n_classes == 3:
            wandb.log(
                {
                    "Validation Average Background Accuracy": vavg_class_acc[0],
                    "Validation Average Cells Accuracy": vavg_class_acc[1],
                    "Validation Average Dendrites Accuracy": vavg_class_acc[2],
                }
            )
            wandb.log(
                {
                    "Training Average Background Accuracy": tavg_class_acc[0],
                    "Training Average Cells Accuracy": tavg_class_acc[1],
                    "Training Average Dendrites Accuracy": tavg_class_acc[2],
                }
            )
            wandb.log(
                {
                    "Validation Average IoU for Background": avg_inter_over_uni[0],
                    "Validation Average IoU for Neurons": avg_inter_over_uni[1],
                    "Validation Average IoU for Dendrites": avg_inter_over_uni[2],
                }
            )
        elif n_classes == 2:
            wandb.log(
                {
                    "Training Background Accuracy": tepoch_class_acc[0],
                    "Training Dendrites Accuracy": tepoch_class_acc[1],
                }
            )
            wandb.log(
                {
                    "Validation Background Accuracy": vepoch_class_acc[0],
                    "Validation Dendrites Accuracy": vepoch_class_acc[1],
                }
            )
            wandb.log(
                {
                    "Validation IoU for Background": inter_over_uni[0],
                    "Validation IoU for Dendrites": inter_over_uni[1],
                }
            )

    try:
        os.mkdir(model_path)
    except OSError:
        pass

    torch.save(net.state_dict(), model_path + model_name)
    if wandb_track:
        torch.save(net.state_dict(), os.path.join(wandb.run.dir, model_name))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_classes = utils.params.n_classes
    n_channels = utils.params.n_channels
    print(
        "Number of input channels = {}. Number of classes = {}.".format(
            n_channels, n_classes
        )
    )
    if n_classes == 3:
        class_weights = np.array([0.3, 0.5, 1]).astype(np.float)
    elif n_classes == 2:
        class_weights = np.array([0.047619, 1]).astype(np.float)
    print("Current class weights = {}".format(class_weights))
    assert (
        len(class_weights) == n_classes
    ), "Should be a 1D Tensor assigning weight to each of the classes. Lenght of the weights-vector should be equal to the number of classes"

    net = UNet(n_channels=n_channels, n_classes=n_classes)
    net.to(device=device)

    try:
        print("Training starting...")
        train_net(net, n_channels, n_classes, class_weights)
        print("Training done")
    except KeyboardInterrupt:
        torch.save(net.state_dict(), model_path + "INTERRUPTED.pth")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
