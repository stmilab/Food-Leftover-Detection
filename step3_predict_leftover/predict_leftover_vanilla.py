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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def initialize_vanilla_model_optim_loss(flags):  # NOTE: the vanilla version
    flags.img_model = flags.img_model.lower()  # lowering all cases
    if flags.img_model == "vit":  # NOTE: ViT model, vanilla feeding 6 images
        vit_model = ViT(
            {
                "embed_dim": crop_size,
                "hidden_dim": 64,
                "num_channels": 18,  # RGB x (Orig+Depth+Segmentation) x (before + after)
                "num_heads": 4,
                "num_layers": 4,
                "num_classes": 64,
                "patch_size": 16,
                "num_patches": (crop_size // 4) ** 2,
                "dropout": 0.1,
            }
        ).to(flags.device)
        img_model = vanilla_vit(vit_model).to(flags.device)
    elif flags.img_model == "cnn":  # NOTE: CNN model
        img_model = CNN(num_parallel=18).to(flags.device)
    else:
        print("Model not recognized")
        raise NotImplementedError
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


def report_loss(predicted, true_labels):
    # Assume these are your predictions and true labels
    # predicted: tensor of shape [batch_size, num_classes]
    # true_labels: tensor of shape [batch_size]

    # Convert logits to class indices
    with torch.no_grad():
        predicted = predicted.cpu()
        true_labels = true_labels.cpu()
        _, predicted_classes = torch.max(predicted, 1)

        # Convert tensors to numpy arrays
        pred_onehot = torch.nn.functional.one_hot(predicted_classes, num_classes=4)
        predicted_classes = predicted_classes.numpy()
        true_labels = true_labels.numpy()
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_classes)
        auroc = roc_auc_score(true_labels, pred_onehot.numpy(), multi_class="ovr")
        precision = precision_score(true_labels, predicted_classes, average="macro")
        recall = recall_score(true_labels, predicted_classes, average="macro")
        f1 = f1_score(true_labels, predicted_classes, average="macro")
        conf_matrix = confusion_matrix(true_labels, predicted_classes)
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "AUROC": auroc,
            "f1": f1,
            "confusion_matrix": conf_matrix,
        }
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"AUROC: {auroc:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        return results


def main():
    shuang.set_random_seed()  # Maintain reproducibility
    flags = leftover_parser()
    # NOTE: Incorporating Weights and Bias
    if flags.use_wandb:
        wandb.init(project="EMBC25 leftover prediction", tags=flags.wandb_tag)
        wandb.config.update(flags)
    log_dict = {}
    # NOTE: loading datasets
    train_set, val_set, test_set = get_dataset(saved_path=flags.saved_path)
    shuang.report_label_distribution(train_set.leftover_label)
    print(
        f"train data: {len(train_set)}; val data: {len(val_set)}; test data: {len(test_set)}"
    )
    # train_sampler = TriSampler(train_set, batch_size_per_task=60, sampling_rate=0.1)
    flags.x_scaler = Sicong_Norm(min_val=0, max_val=255)  # NOTE: image normalizer
    train_loader = DataLoader(
        train_set,
        batch_size=flags.batch_size,
        # sampler=train_sampler,
        shuffle=False,
    )
    # val_sampler = TriSampler(val_set, batch_size_per_task=10, sampling_rate=0.1)
    val_loader = DataLoader(
        val_set,
        batch_size=flags.batch_size,
        # sampler=val_sampler,
        shuffle=False,
    )
    # test_sampler = TriSampler(test_set, batch_size_per_task=10, sampling_rate=0.1)
    test_loader = DataLoader(
        test_set,
        batch_size=9999,
        # sampler=test_sampler,
        shuffle=False,
    )
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
            predicted = img_model(scaled_img.to(flags.device))
            test_loss = criterion(predicted, label_onehot.to(flags.device))
            print(f"TEST LOSS is: {test_loss:.4f}")
    log_dict["loss type"] = flags.loss_type  # logging information
    log_dict["train loss"] = train_loss_list  # adding training information
    log_dict["val loss"] = val_loss_list  # adding validation information
    log_dict.update(report_loss(predicted, label))  # logging results
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
