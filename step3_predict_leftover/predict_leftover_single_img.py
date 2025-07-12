import sys, os, random
from tqdm import tqdm
import numpy as np
import torch
import pdb
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
import torch.nn as nn
import wandb  # Weights and Bias for logging and experiments
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# NOTE: internal imports
from utils.data_loader import get_dataset, LeftoverDataset, crop_size
from utils.parser import leftover_parser
from models.vgg import VGG
from models.cnn import CustomCNN as CNN
from models.vit import ViT
from models.exchange import vanilla_vit
from utils.helper import Sicong_Norm
import cv2

import predict_leftover_cen_vit as shuang

"""
This approach takes in just 1 image to predict the leftover
"""


def initialize_vanilla_model_optim_loss(flags):  # NOTE: the vanilla version
    flags.img_model = flags.img_model.lower()  # lowering all cases
    # Feeding in all channels without any reorganization
    if flags.img_model == "vit":
        img_model = ViT(
            {
                "embed_dim": crop_size,
                "hidden_dim": flags.hidden_dim,
                "num_channels": 3,  # 9 channels for the stacked image
                "num_heads": 4,
                "num_layers": 4,
                "num_classes": 4,  # NOTE: 4 classes for the leftover prediction
                "patch_size": 16,
                "num_patches": (crop_size // 4) ** 2,
                "dropout": flags.dropout_rate,
            }
        ).to(flags.device)
    elif flags.img_model == "cnn":
        img_model = CNN(num_parallel=18).to(flags.device)
    optimizer = optim.Adam(
        img_model.parameters(),
        lr=flags.lr,
        weight_decay=flags.weight_decay,
    )
    if flags.loss_type == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    else:
        print("Loss type not recognized")
        raise NotImplementedError
    return img_model, optimizer, criterion


def main():
    flags, train_set, val_set, test_set, log_dict = shuang.initialize_model()
    train_loader, val_loader, test_loader = shuang.get_dataloaders(
        train_set, val_set, test_set, flags
    )  # NOTE: getting dataloaders
    # NOTE: Initializing model and loss
    img_model, optimizer, criterion = initialize_vanilla_model_optim_loss(flags)
    # img_model = nn.DataParallel(img_model)
    # NOTE: Training and validating the model
    train_loss_list, val_loss_list = [], []
    for epoch in tqdm(range(flags.epochs), ascii=True, desc="Training"):
        # NOTE: Training
        img_model.train()
        train_avg_loss = []
        for (b_img, a_img), label in train_loader:
            # Forward pass
            if flags.loss_type == "CrossEntropy":
                label_onehot = label
            else:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
            a_img = a_img.view(-1, 9, crop_size, crop_size)
            b_img = b_img.view(-1, 9, crop_size, crop_size)
            stacked_img = torch.cat((b_img, a_img), dim=1)
            scaled_img = flags.x_scaler.norm(stacked_img)
            predicted = img_model(scaled_img.to(flags.device))
            train_loss = criterion(predicted, label_onehot.to(flags.device))
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_avg_loss.append(train_loss.item())
        train_loss_list.append(np.array(train_avg_loss).mean())
        # NOTE: Validation
        img_model.eval()
        val_avg_loss = []
        with torch.no_grad():
            for (b_img, a_img), label in val_loader:
                # Forward pass
                if flags.loss_type == "CrossEntropy":
                    label_onehot = label
                else:
                    label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
                a_img = a_img.view(-1, 9, crop_size, crop_size)
                b_img = b_img.view(-1, 9, crop_size, crop_size)
                stacked_img = torch.cat((b_img, a_img), dim=1)
                scaled_img = flags.x_scaler.norm(stacked_img)
                predicted = img_model(scaled_img.to(flags.device))
                val_loss = criterion(predicted, label_onehot.to(flags.device))
                val_avg_loss.append(val_loss.item())
            val_loss_list.append(np.array(val_avg_loss).mean())
            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch: {epoch+1}; train loss: {train_loss_list[-1]:.4f}; val loss: {val_loss_list[-1]:.4f}"
                )

    # NOTE: Evaluating trained model
    img_model.eval()
    with torch.no_grad():
        img_model = img_model.cpu()  # NOTE: moving model to CPU for evaluation
        for (b_img, a_img), label in test_loader:
            # Forward pass
            if flags.loss_type == "CrossEntropy":
                label_onehot = label
            else:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
            a_img = a_img.view(-1, 9, crop_size, crop_size)
            b_img = b_img.view(-1, 9, crop_size, crop_size)
            stacked_img = torch.cat((b_img, a_img), dim=1)
            scaled_img = flags.x_scaler.norm(stacked_img)
            predicted = img_model(scaled_img)
            test_loss = criterion(predicted, label_onehot)
            print(f"TEST LOSS is: {test_loss:.4f}")
    log_dict["loss type"] = flags.loss_type  # logging information
    log_dict["train loss"] = train_loss_list  # adding training information
    log_dict["val loss"] = val_loss_list  # adding validation information
    log_dict.update(shuang.report_loss(predicted, label))  # logging results
    # visuals = shuang.get_visual(
    #     img_model,
    #     test_loader,
    #     predicted,
    #     save_path=f"predicted_imgs/vanilla_{flags.img_model}/",
    # )  # visualizing the results
    if flags.use_wandb:
        wandb.log(log_dict)
        wandb.join()
    # pdb.set_trace()


if __name__ == "__main__":
    main()
