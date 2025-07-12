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
from utils.data_loader_bsn25 import get_balanced_dataset
from models.vgg import VGG
from models.vit import ViT
from models.cnn import CustomCNN as CNN
from models.exchange import CEN, CEN_CNN
from utils.helper import Sicong_Norm
import cv2
import seaborn as sns

# NOTE: libAUC imports, deprecated
# import libauc # Adapting AUC loss
# from libauc.sampler import TriSampler
# from libauc.losses import MultiLabelAUCMLoss, MultiLabelpAUCLoss, AUCMLoss

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


leftover_mapping = {
    0: "No Leftover: 0~25%",
    1: "Little Leftover: 25~50%",
    2: "Some Leftover: 50~75%",
    3: "Full: 75~100%",
}


def attention_rollout(attentions):
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    # Multiply attention maps layer by layer
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1)  # Average attention across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(
            attention_heads_fused.device
        )  # A + I
        attention_heads_fused /= attention_heads_fused.sum(
            dim=-1, keepdim=True
        )  # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused)  # Multiplication

    return rollout


# def visualize_attention_map(attn_maps, img):
#     """
#     Visualize the attention1 map.

#     Parameters:
#     - attn_maps: List of attention maps from each layer.
#     - img: The original image tensor.
#     - layer_idx: Index of the layer to visualize. Default is the last layer.
#     - head_idx: Index of the head to visualize. Default is the average of all heads.
#     """
#     # Select the attention map from the specified layer
#     exit
#     att_mat = torch.stack(attn_maps).squeeze(1)

#     # Average the attention weights across all heads if head_idx is -1
#     att_mat = att_mat.mean(dim=1)
#     # To account for residual connections, we add an identity matrix to the
#     # attention matrix and re-normalize the weights.
#     residual_att = torch.eye(att_mat.size(-1))

#     aug_att_mat = att_mat + residual_att
#     aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

#     # Recursively multiply the weight matrices
#     joint_attentions = torch.zeros(aug_att_mat.size())
#     joint_attentions[0] = aug_att_mat[0]

#     for n in range(1, aug_att_mat.size(0)):
#         joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

#     v = joint_attentions[-1]
#     grid_size = int(np.sqrt(aug_att_mat.size(-1)))
#     mask = v[0, 1:].reshape(grid_size, grid_size)
#     # Resize the mask to the size of the original image
#     mask = cv2.resize(mask / mask.max(), img.size)
#     result = (mask * img).astype("uint8")

#     # Plot the attention map
#     plt.close()
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(img.permute(1, 2, 0).cpu().numpy())
#     ax[0].set_title("Original Image")
#     ax[0].axis("off")
#     ax[1].imshow(img.permute(1, 2, 0).cpu().numpy())
#     ax[1].imshow(mask, cmap="viridis", alpha=0.6)
#     ax[1].set_title("Attention Map")
#     ax[1].axis("off")

#     # Save the figure
#     plt.savefig("attention_map.png")
#     plt.close()
#     return result


def label_noise(labels, noise_level=0.2):
    """
    Add random noise to the labels.

    Parameters:
    - labels: List of labels to add noise to.
    - noise_level: The proportion of labels to change.

    Returns:
    - List of labels with noise added.
    """
    with torch.no_grad():
        noisy_labels = labels.clone()
        num_noisy_labels = int(len(labels) * noise_level)

        # Randomly select indices to change
        noisy_indices = random.sample(range(len(labels)), num_noisy_labels)

        # Change the selected labels
        for idx in noisy_indices:
            noisy_labels[idx] = random.choice(
                [label for label in range(4) if label != labels[idx]]
            )
    return noisy_labels


def plot_one_img(img_arr: torch.tensor, dest: str = "plot_one_img.png"):
    img_arr = img_arr.int().permute(1, 2, 0).numpy()
    plt.close()
    plt.imshow(img_arr)
    plt.axis("off")
    plt.savefig(dest)
    plt.close()


def set_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor2img(tensor):
    # return transforms.ToPILImage(mode="RGB")(tensor) # color issue
    return tensor.int().permute(1, 2, 0).numpy()


def get_visual(model, test_loader, predicted, save_path="predicted_imgs_cen_vit/"):
    model.eval()
    with torch.no_grad():
        # transform = transforms.ToPILImage()
        # cen_model = model.module.cpu()  # sending model to CPU
        cen_model = model.cpu()  # sending model to CPU
        # Select a few random samples
        # NOTE: visualizing the prediction - randomly picking 5
        indices = np.random.choice(len(test_loader.dataset), 10, replace=False)
        fontsize = 16
        for idx in indices:
            # NOTE: the [i, 0] equals [i], the second element doesn't present any meaning in this context, just to incorporate the modified dataset getitem method, according to libAUC's sampler
            (b_img, a_img), label = test_loader.dataset[idx]
            orig_before, dep_before, seg_before = b_img[0], b_img[1], b_img[2]
            orig_after, dep_after, seg_after = a_img[0], a_img[1], a_img[2]
            predicted = cen_model(b_img.unsqueeze(0), a_img.unsqueeze(0))
            pred_class = predicted.argmax(dim=1).item()
            label_class = label.item()
            # get_attn = cen_model.before_vit.model  # getting attention map, not used
            # b_attn = get_attn(b_img, return_attn=True)[1]
            # visualize_attention_map(b_attn, orig_before)
            fig, axs = plt.subplots(2, 3, figsize=(12, 12), dpi=100)
            # Plot before image - orig
            axs[0, 0].imshow(tensor2img(orig_before))
            axs[0, 0].set_title(f"Image before meal\n (original)", fontsize=fontsize)
            axs[0, 0].axis("off")

            # Plot before image - dep
            axs[0, 2].imshow(tensor2img(dep_before))
            axs[0, 2].set_title("Image before meal\n (depth)", fontsize=fontsize)
            axs[0, 2].axis("off")

            # Plot before image - seg
            axs[0, 1].imshow(tensor2img(seg_before))
            axs[0, 1].set_title("Image before meal\n (segmentation)", fontsize=fontsize)
            axs[0, 1].axis("off")

            # Plot after image - orig
            axs[1, 0].imshow(tensor2img(orig_after))
            axs[1, 0].set_title("Image after meal\n (original)", fontsize=fontsize)
            axs[1, 0].axis("off")

            # Plot after image - dep
            axs[1, 2].imshow(tensor2img(dep_after))
            axs[1, 2].set_title("Image after meal\n (depth)", fontsize=fontsize)
            axs[1, 2].axis("off")

            # Plot after image - seg
            axs[1, 1].imshow(tensor2img(seg_after))
            axs[1, 1].set_title("Image after meal\n (segmentation)", fontsize=fontsize)
            axs[1, 1].axis("off")
            plt.tight_layout()
            fig.suptitle(
                f"predicted: {leftover_mapping[pred_class]}\n"
                + f"ground truth: {leftover_mapping[label_class]}",
                fontsize=fontsize,
            )
            if not os.path.exists(save_path):  # making the directory is not exist
                os.makedirs(save_path)
            plt.savefig(f"{save_path}attn_map_sample_{idx}.png")
            plt.close()
    print("VISUALIZATION DONE!")
    return


def report_loss(predicted, true_labels):
    # Assume these are your predictions and true labels
    # predicted: tensor of shape [batch_size, num_classes]
    # true_labels: tensor of shape [batch_size]

    # Convert logits to class indices
    with torch.no_grad():
        _, predicted_classes = torch.max(predicted, 1)

        # Convert tensors to numpy arrays
        true_labels = true_labels.cpu().numpy()
        pred_onehot = torch.nn.functional.one_hot(predicted_classes, num_classes=4)
        predicted_classes = predicted_classes.cpu().numpy()
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_classes)
        auroc = roc_auc_score(true_labels, pred_onehot.cpu().numpy(), multi_class="ovr")
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


def plot_confusion_matrix_from_cm(cm, class_names=None, normalize=False, cmap="Blues"):
    cm = np.array(cm)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        ax=ax,
        xticklabels=class_names if class_names is not None else range(cm.shape[1]),
        yticklabels=class_names if class_names is not None else range(cm.shape[0]),
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig("confusion_matrix.png")
    return fig


def initialize_model_optim_loss(flags):
    flags.img_model = flags.img_model.lower()  # lowering all cases
    flags.cen_mode = flags.cen_mode.lower()  # lowering all cases
    if flags.img_model == "vit":
        img_model = [
            ViT(
                {
                    "embed_dim": crop_size,
                    "hidden_dim": flags.hidden_dim,
                    "num_channels": 3,  # RGB image
                    "num_heads": 4,
                    "num_layers": 4,
                    "num_classes": 64,
                    "patch_size": 16,
                    "num_patches": (crop_size // 4) ** 2,
                    "dropout": flags.dropout_rate,
                }
            )
            for _ in range(2)
        ]
    elif flags.img_model == "vgg":
        img_model = [VGG() for _ in range(2)]
    elif flags.img_model == "cnn":
        img_model = [CNN(cen_mode=True) for _ in range(2)]
    else:
        print("image model not recognized")
        raise NotImplementedError
    if flags.cen_mode == "cen_cnn":
        cen_model = CEN_CNN(
            threshold=flags.cen_threshold,
            crop_size=crop_size,
        ).to(flags.device)
    elif flags.cen_mode == "cen":
        cen_model = CEN(
            img_model[0],
            img_model[1],
            threshold=flags.cen_threshold,
            crop_size=crop_size,
        ).to(flags.device)
    else:
        print("CEN mode not recognized")
        raise NotImplementedError
    optimizer = optim.Adam(
        cen_model.parameters(),
        lr=flags.lr,
        weight_decay=flags.weight_decay,
    )
    if flags.loss_type == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    # elif flags.loss_type=="AUC": # Multilabel AUC loss - adoped f rom Dr. Yang's lab
    #     criterion = MultiLabelAUCMLoss(num_labels=3, device=flags.device)
    else:
        print("Loss type not recognized")
        raise NotImplementedError
    return cen_model, optimizer, criterion


def report_label_distribution(labels):
    unique, counts = torch.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    # Ensure all labels (0, 1, 2, 3) are in the dictionary
    for label in range(3):
        if label not in label_counts:
            label_counts[label] = 0
    for label, count in label_counts.items():
        print(f"Label {leftover_mapping[label]}: {count} occurrences")


def initialize_model():
    set_random_seed()  # Maintain reproducibility
    flags = leftover_parser()
    # NOTE: Incorporating Weights and Bias
    if flags.use_wandb:
        wandb.init(project="EMBC25 leftover prediction", tags=[flags.wandb_tag])
        wandb.config.update(flags)
    log_dict = {}
    # NOTE: loading datasets
    # train_set, val_set, test_set = get_dataset(saved_path=flags.saved_path)
    train_set, val_set, test_set = get_balanced_dataset(
        saved_dataset_fname=flags.saved_path
    )
    print("train")
    report_label_distribution(train_set.leftover_label)
    print("val")
    report_label_distribution(val_set.leftover_label)
    print("test")
    report_label_distribution(test_set.leftover_label)
    print(
        f"train data: {len(train_set)}; val data: {len(val_set)}; test data: {len(test_set)}"
    )
    flags.x_scaler = Sicong_Norm(min_val=0, max_val=255)  # NOTE: image normalizer
    return flags, train_set, val_set, test_set, log_dict


def get_dataloaders(train_set, val_set, test_set, flags):
    # train_sampler = TriSampler(train_set, batch_size_per_task=60, sampling_rate=0.1)
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
    return train_loader, val_loader, test_loader


def main():
    flags, train_set, val_set, test_set, log_dict = initialize_model()
    train_loader, val_loader, test_loader = get_dataloaders(
        train_set, val_set, test_set, flags
    )  # NOTE: getting dataloaders
    # NOTE: Initializing model and loss
    cen_model, optimizer, criterion = initialize_model_optim_loss(flags)
    # cen_model = nn.DataParallel(cen_model)
    # NOTE: Training and validating the model
    train_loss_list, val_loss_list = [], []
    for epoch in tqdm(range(flags.epochs), ascii=True, desc="Training"):
        # NOTE: Training
        cen_model.train()
        train_avg_loss = []
        for (b_img, a_img), label in train_loader:
            # Forward pass
            if flags.loss_type == "CrossEntropy":
                label_onehot = label
            else:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
            scaled_b, scaled_a = flags.x_scaler.norm(b_img), flags.x_scaler.norm(a_img)
            predicted = cen_model(scaled_b.to(flags.device), scaled_a.to(flags.device))
            train_loss = criterion(predicted, label_onehot.to(flags.device))
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_avg_loss.append(train_loss.item())
        train_loss_list.append(np.array(train_avg_loss).mean())
        # NOTE: Validation
        cen_model.eval()
        val_avg_loss = []
        with torch.no_grad():
            for (b_img, a_img), label in val_loader:
                # Forward pass
                if flags.loss_type == "CrossEntropy":
                    label_onehot = label
                else:
                    label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
                scaled_b, scaled_a = flags.x_scaler.norm(b_img), flags.x_scaler.norm(
                    a_img
                )
                predicted = cen_model(
                    scaled_b.to(flags.device), scaled_a.to(flags.device)
                )
                val_loss = criterion(predicted, label_onehot.to(flags.device))
                val_avg_loss.append(val_loss.item())
            val_loss_list.append(np.array(val_avg_loss).mean())
            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch: {epoch+1}; train loss: {train_loss_list[-1]:.4f}; val loss: {val_loss_list[-1]:.4f}"
                )

    # NOTE: Evaluating trained model
    cen_model.eval()
    with torch.no_grad():
        cen_model = cen_model.cpu()  # NOTE: moving model to CPU for evaluation
        for (b_img, a_img), label in test_loader:
            # Forward pass
            label = label_noise(label)
            if flags.loss_type == "CrossEntropy":
                label_onehot = label
            else:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
            scaled_b, scaled_a = flags.x_scaler.norm(b_img), flags.x_scaler.norm(a_img)
            predicted = cen_model(scaled_b, scaled_a)
            test_loss = criterion(predicted, label_onehot)
            print(f"TEST LOSS is: {test_loss:.4f}")
    log_dict["loss type"] = flags.loss_type  # logging information
    log_dict["train loss"] = train_loss_list  # adding training information
    log_dict["val loss"] = val_loss_list  # adding validation information
    log_dict.update(report_loss(predicted, label))  # logging results
    fig = plot_confusion_matrix_from_cm(
        log_dict["confusion_matrix"], class_names=list(leftover_mapping.values())
    )
    visuals = get_visual(
        cen_model,
        test_loader,
        predicted,
        save_path=f"predicted_imgs/cen_{flags.img_model}/",
    )  # visualizing the results
    if flags.use_wandb:
        wandb.log(log_dict)
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        wandb.join()
    else:

        pdb.set_trace()


if __name__ == "__main__":
    main()
