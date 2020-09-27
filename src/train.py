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

data_folder = "../data/"
model_path = "../model/"

def train_net(
    net,n_channels,n_classes, epochs=100, val_precent=0.1, batch_size=1, lr=0.0001, weight_decay=1e-8, momentum=0.99
):
    print("Creating dataset for training...")
    dataset = Loader(data_folder)
    n_val = int(len(dataset) * val_precent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    global_step = 0

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor(class_weights).to(device=device)
    )

    for epoch in range(epochs):
        net.train()
        tepoch_loss = 0
        tepoch_acc = 0
        vepoch_loss = 0
        vepoch_acc = 0
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
            deb = False
            if deb:
                print(masks_pred.shape)
            loss = criterion(masks_pred, masks.squeeze(1))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch_loss += loss.item()
            tepoch_acc += multi_acc(masks_pred, masks) 

            global_step += 1

        net.eval()
        for batch in val_loader:
            with torch.no_grad():
                imgs = batch["image"]
                masks = batch["mask"]
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, masks.squeeze(1))

                vepoch_loss += loss.item() 
                vepoch_acc += multi_acc(masks_pred, masks)

        tepoch_loss /= n_train
        tepoch_acc /= n_train
        vepoch_loss /= n_val
        vepoch_acc /= n_val

        print(
            "epoch {0:} finished, tloss: {1:.4f} [{2:.2f}%]  vloss: {3:.4f} [{4:.2f}%]".format(
                epoch + 1, tepoch_loss, tepoch_acc, vepoch_loss, vepoch_acc
            )
        )
    try:
        os.mkdir(model_path)
    except OSError:
        pass
    torch.save(net.state_dict(), model_path + "model.pth")


def multi_acc(pred, label):
    _, tags = torch.max(pred, dim=1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = np.array([1, 1, 1]).astype(np.float)
    n_classes = 3
    n_channels = 3
    assert (
        len(class_weights) == n_classes
    ), "Lenght of the weights-vector should be equal to the number of classes"

    net = UNet(n_channels=n_channels,n_classes=n_classes)
    net.to(device=device)

    try:
        print("Training starting...")
        train_net(net,n_channels,n_classes)
        print("Training done")
    except KeyboardInterrupt:
        torch.save(net.state_dict(), model_path + "INTERRUPTED.pth")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
