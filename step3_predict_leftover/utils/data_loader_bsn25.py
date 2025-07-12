"""
This is a variation to process the balanced dataset made by Mahima
"""

import numpy as np
import json
import pdb
import pandas as pd
from tqdm import tqdm
import random
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import sys

# from libauc.sampler import TriSampler
from data_loader import (
    set_random_seed,
    load_meta_data,
    generate_split,
    leftover_label_convert,
    iterate_orig_dataframe,
    LeftoverDataset,
)

sys.path.append("utils/")
sys.path.append("models/")

crop_size = 224


def iterate_augmented_dataframe(aug_df: pd.DataFrame, merge_dataset=None) -> Dataset:
    if "augmented" not in aug_df["image type"].unique():
        raise ValueError("aug_df should at least contain 1 augmented image type")
    leftover_dataset = LeftoverDataset(len(aug_df))
    for _, row in tqdm(aug_df.iterrows(), total=len(aug_df), ascii=True):
        orig_before, dep_before, seg_before = get_augmented_img(row["before image"])
        orig_after, dep_after, seg_after = get_augmented_img(row["after image"])

        before_img = np.stack((orig_before, dep_before, seg_before), axis=0)
        after_img = np.stack((orig_after, dep_after, seg_after), axis=0)
        leftover_label = row["leftover"]
        leftover_dataset.append(before_img, after_img, leftover_label)
    leftover_dataset.convert_to_torch()
    if merge_dataset is not None:
        print("Merging augmented dataset with existing dataset")
        leftover_dataset.merge_dataset(merge_dataset)
    return leftover_dataset


def load_augmented_dataset(
    augmented_dir="/home/mahima19/augmented_dataset_global_torch_new/",
):
    """
    load_augmented_dataset Loading the augmented dataset, does not change original dataframe

    Args:
        augmented_dir (str, optional): directory of augmented images. Defaults to "/home/mahima19/augmented_dataset_global_torch_new/".
    """
    # NOTE: example: "1/49_b_orig_t1.jpg" -> label 1 (directory), 49 th meal, orig image , task 1
    before_img_dict = {
        "nametag": [],
        "before_or_after": [],
        "image_fname": [],
        "leftover": [],
    }

    after_img_dict = {
        "nametag": [],
        "before_or_after": [],
        "image_fname": [],
        "leftover": [],
    }
    for subdir, _, fnames in os.walk(augmented_dir):
        for fname in fnames:
            if fname.endswith(".jpg") or fname.endswith(".png"):
                # Extracting meal type and meal id from the filename
                s_parts = fname[:-4].split("_")
                before_or_after = s_parts[1]  # before or after
                if before_or_after == "b":
                    new_dict = before_img_dict
                elif before_or_after == "a":
                    new_dict = after_img_dict
                else:
                    raise ValueError(
                        f"Unexpected before_or_after value: {before_or_after}"
                    )
                new_dict["nametag"].append(f"{s_parts[0]}_{s_parts[2]}_{s_parts[3]}")
                new_dict["before_or_after"].append(before_or_after)
                new_dict["image_fname"].append(os.path.join(subdir, fname))
                new_dict["leftover"].append(int(subdir[-1]))
    before_df = pd.DataFrame(before_img_dict)
    after_df = pd.DataFrame(after_img_dict)
    # Merging the two dataframes
    merged_df = pd.merge(
        before_df,
        after_df,
        on=["nametag", "leftover"],
        suffixes=("_before", "_after"),
    )
    merged_df.rename(
        columns={
            "image_fname_before": "before image",
            "image_fname_after": "after image",
        },
        inplace=True,
    )
    merged_df.drop(
        columns=["before_or_after_before", "before_or_after_after"],
        inplace=True,
    )

    return merged_df


def get_balanced_dataset(
    regenerate: bool = False,
    saved_dataset_fname: str = "balanced_dataset.pth",
) -> Dataset:
    """
    get_balanced_dataset Generating the leftover dataset

    Args:
        regenerate (bool, optional): Whether to generate from scratch. Defaults to False.
        saved_dataset_fname (str, optional): Path to saved dataset. Defaults to "balanced_dataset.pth".

    Returns:
        Dataset: Generated dataset
    """
    if not os.path.exists(saved_dataset_fname):
        regenerate = True  # if no found, then generate anyway
    if regenerate:
        # NOTE: Loading the original dataset
        orig_img_df = load_meta_data()
        orig_img_df["leftover"] = orig_img_df["leftover"].apply(leftover_label_convert)
        orig_img_df["image type"] = "original"
        train_orig_df, val_orig_df, test_orig_df = generate_split(orig_img_df)
        print("loading original dataset")
        if os.path.exists("dataset.pth"):
            train_dataset, val_dataset, test_dataset = torch.load("dataset.pth")
        else:
            train_dataset = iterate_orig_dataframe(train_orig_df)
            val_dataset = iterate_orig_dataframe(val_orig_df)
            test_dataset = iterate_orig_dataframe(test_orig_df)
        # NOTE: Loading the augmented dataset
        aug_df = load_augmented_dataset()
        aug_df["image type"] = "augmented"
        train_aug_df, val_aug_df, test_aug_df = generate_split(aug_df)
        # NOTE: Merging the two dataframes
        print("loading augmented dataset")
        train_dataset = iterate_augmented_dataframe(train_aug_df, train_dataset)
        val_dataset = iterate_augmented_dataframe(val_aug_df, val_dataset)
        # test_dataset = iterate_augmented_dataframe(test_aug_df, test_dataset)
        # Create dataset
        # FIXME: Start counting - update them after merging
        torch.save((train_dataset, val_dataset, test_dataset), saved_dataset_fname)
        print(f"Balanced Leftover Dataset saved to {saved_dataset_fname}!")
        return train_dataset, val_dataset, test_dataset
    return torch.load(saved_dataset_fname)  # otherwise return pre-saved


def get_augmented_img(
    aug_img_fullname: str,
    aug_seg_dir="/home/mahima19/dino_segmented_images/",
    aug_depth_dir="/home/mahima19/dino_depth/",
    target_size=(crop_size, crop_size),
):
    """
    get_augmented_img Loads and returns the original, depth, and segmentation images for a given augmented image filename, resizing them to the specified target size.

    Args:
        aug_img_fullname (str): Full path to the augmented image file.
        aug_seg_dir (str, optional): Directory containing segmentation images. Defaults to "/home/mahima19/dino_segmented_images/".
        aug_depth_dir (str, optional): Directory containing depth images. Defaults to "/home/mahima19/dino_depth/".
        target_size (tuple, optional): Target size (width, height) for resizing images. Defaults to (crop_size, crop_size).
    """
    # NOTE: Constructing the full path for depth and segmentation images
    aug_dirname, base_fname = os.path.split(aug_img_fullname)
    base_fname = base_fname[:-4]
    depth_img_dirname = aug_depth_dir + aug_dirname[-1]
    depth_img_fullname = os.path.join(depth_img_dirname, base_fname + "_depth.png")
    seg_img_dirname = aug_seg_dir + aug_dirname[-1]
    seg_img_fullname = os.path.join(seg_img_dirname, base_fname + "_segmented.png")
    # NOTE: Loading names out
    orig_img = cv2.resize(
        cv2.cvtColor(cv2.imread(aug_img_fullname), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )
    dep_img = cv2.resize(
        cv2.cvtColor(cv2.imread(depth_img_fullname), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )
    seg_img = cv2.resize(
        cv2.cvtColor(cv2.imread(seg_img_fullname), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )
    return orig_img, dep_img, seg_img


if __name__ == "__main__":
    set_random_seed()
    train_set, val_set, test_set = get_balanced_dataset(regenerate=True)
    # Create DataLoader based on the saved dataset
    # train_sampler = TriSampler(train_set, batch_size_per_task=10, sampling_rate=0.1)
    train_loader = DataLoader(
        train_set,
        batch_size=9999,
        # sampler=train_sampler,
        shuffle=False,
    )
    # Iterate through the DataLoader
    for data, leftover_label in train_loader:
        before_img, after_img = data
        # torch.Size([300, 3, crop_size, crop_size, 3]) torch.Size([300, 3, crop_size, crop_size, 3]) torch.Size([300])
        print(before_img.shape, after_img.shape, leftover_label.shape)
    # val_sampler = TriSampler(val_set, batch_size_per_task=10, sampling_rate=0.1)
    val_dataloader = DataLoader(
        val_set,
        batch_size=9999,
        # sampler=val_sampler,
        shuffle=False,
    )
    for data, leftover_label in val_dataloader:
        before_img, after_img = data
        # torch.Size([300, 3, crop_size, crop_size, 3]) torch.Size([300, 3, crop_size, crop_size, 3]) torch.Size([300])
        print(before_img.shape, after_img.shape, leftover_label.shape)
    # test_sampler = TriSampler(test_set, batch_size_per_task=10, sampling_rate=0.1)
    test_dataloader = DataLoader(
        test_set,
        batch_size=9999,
        # sampler=test_sampler,
        shuffle=False,
    )
    for data, leftover_label in test_dataloader:
        before_img, after_img = data
        # torch.Size([300, 3, crop_size, crop_size, 3]) torch.Size([300, 3, crop_size, crop_size, 3]) torch.Size([300])
        print(before_img.shape, after_img.shape, leftover_label.shape)
    print("Completed generating dataset")
    # import torchvision.transforms as transforms
    # import matplotlib.pyplot as plt

    # def tensor2img(tensor):
    #     return tensor.int().permute(1, 2, 0).numpy()

    # def visual_todo(idx):
    #     (b_img, a_img), label = test_dataloader.dataset[idx]
    #     orig_before, dep_before, seg_before = b_img[0], b_img[1], b_img[2]
    #     orig_after, dep_after, seg_after = a_img[0], a_img[1], a_img[2]
    #     fig, axs = plt.subplots(2, 3, figsize=(10, 9))
    #     # Plot before image - orig
    #     axs[0, 0].imshow(tensor2img(orig_before))
    #     axs[0, 0].set_title(f"Image before meal (original)")
    #     axs[0, 0].axis("off")

    #     # Plot before image - dep
    #     axs[0, 1].imshow(tensor2img(dep_before))
    #     axs[0, 1].set_title("Image before meal (depth)")
    #     axs[0, 1].axis("off")

    #     # Plot before image - seg
    #     axs[0, 2].imshow(tensor2img(seg_before))
    #     axs[0, 2].set_title("Image before meal (segmentation)")
    #     axs[0, 2].axis("off")

    #     # Plot after image - orig
    #     axs[1, 0].imshow(tensor2img(orig_after))
    #     axs[1, 0].set_title("Image after meal (original)")
    #     axs[1, 0].axis("off")

    #     # Plot after image - dep
    #     axs[1, 1].imshow(tensor2img(dep_after))
    #     axs[1, 1].set_title("Image after meal (depth)")
    #     axs[1, 1].axis("off")

    #     # Plot after image - seg
    #     axs[1, 2].imshow(tensor2img(seg_after))
    #     axs[1, 2].set_title("Image after meal (segmentation)")
    #     axs[1, 2].axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f"../stupid_sicong.png")
    #     plt.close()

    # pdb.set_trace()
