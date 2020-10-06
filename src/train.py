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

wandb_track = False
if wandb_track:
    import wandb

    wandb.init(project="unet-med")

data_folder = "../data/"
model_path = "../model/"
model_name = "exp.pth"


def train_net(
    net,
    n_channels,
    n_classes,
    class_weights,
    epochs=1,
    val_precent=0.1,
    batch_size=1,
    lr=0.0001,
    weight_decay=1e-8,
    momentum=0.99,
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

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(
        weight=torch.Tensor(class_weights).to(device=device)
    )
    if wandb_track:
        wandb.watch(net)
    for epoch in range(epochs):
        net.train()
        tepoch_loss = 0
        tepoch_acc = 0
        vepoch_loss = 0
        vepoch_acc = 0
        tepoch_class_acc = [0] * n_classes
        vepoch_class_acc = [0] * n_classes

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
            loss = criterion(masks_pred, masks.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch_loss += loss.item()
            tepoch_acc += multi_acc(masks_pred, masks)
            tepoch_class_acc = np.add(np.array(tepoch_class_acc),np.array(multi_acc_class(masks_pred, masks, n_classes)))

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
                vepoch_class_acc = np.add(np.array(vepoch_class_acc),np.array(multi_acc_class(masks_pred, masks, n_classes)))

        tepoch_loss /= n_train
        tepoch_acc /= n_train
        tepoch_class_acc = [x / n_train for x in tepoch_class_acc]
        vepoch_loss /= n_val
        vepoch_acc /= n_val
        vepoch_class_acc = [x / n_val for x in vepoch_class_acc]

        print(
            "Epoch {0:}, Training loss: {1:.4f} [{2:.2f}%]  Validation loss: {3:.4f} [{4:.2f}% ".format(
                epoch + 1,
                tepoch_loss,
                tepoch_acc,
                vepoch_loss,
                vepoch_acc,
            )
        )
        print(
            "Training Class 1 Accuracy: [{0:.2f}%] Training Class 2 Accuracy: [{1:.2f}%] Training Class 3 Accuracy: [{2:.2f}%]".format(
                tepoch_class_acc[0],
                tepoch_class_acc[1],
                tepoch_class_acc[2],
            )
        )
        print(
            "Validation Class 1 Accuracy: [{0:.2f}%] Validation Class 2 Accuracy: [{1:.2f}%] Validation Class 3 Accuracy: [{2:.2f}%]".format(
                vepoch_class_acc[0],
                vepoch_class_acc[1],
                vepoch_class_acc[2],
            )
        )
        if wandb_track:
            wandb.log({"Test Accuracy": tepoch_acc, "Test Loss": tepoch_loss})
            wandb.log(
                {"Validation Accuracy": vepoch_acc, "Validation Loss": vepoch_loss}
            )
    try:
        os.mkdir(model_path)
    except OSError:
        pass

    torch.save(net.state_dict(), model_path + model_name)
    if wandb_track:
        torch.save(net.state_dict(), os.path.join(wandb.run.dir, model_name))


def multi_acc(pred, label):
    tags = torch.argmax(pred, dim=1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


def multi_acc_class(pred, label, n_classes):
    accs_per_label_pct = []
    tags = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        corrects = label == cls
        num_total_per_label = corrects.sum()
        corrects &= tags == label
        num_corrects_per_label = corrects.float().sum()
        accs_per_label_pct.append(num_corrects_per_label / num_total_per_label * 100)
    return [i.item() for i in accs_per_label_pct]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 3
    n_channels = 3
    print(
        "Number of input channels = {}. Number of classes = {}".format(
            n_channels, n_classes
        )
    )
    class_weights = np.array([0.3, 0.5, 1]).astype(np.float)
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
