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
from sklearn.model_selection import train_test_split

# from libauc.sampler import TriSampler

sys.path.append("utils/")
sys.path.append("models/")

crop_size = 224


def set_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iterate_dataframe(df):
    dataset = LeftoverDataset(len(df))
    for i, row in tqdm(df.iterrows(), total=len(df), ascii=True):
        orig_before, dep_before, seg_before = get_img(row["before image"])
        orig_after, dep_after, seg_after = get_img(row["after image"])

        before_img = np.stack((orig_before, dep_before, seg_before), axis=0)
        after_img = np.stack((orig_after, dep_after, seg_after), axis=0)
        leftover_label = leftover_label_convert(row["leftover"])
        dataset.append(before_img, after_img, leftover_label)
    dataset.convert_to_torch()
    return dataset


def get_dataset(regenerate: bool = False, saved_path: str = "dataset.pth") -> Dataset:
    """
    get_dataset Generating the leftover dataset

    Args:
        regenerate (bool, optional): Whether to generate from scratch. Defaults to False.
        saved_path (str, optional): Path to saved dataset. Defaults to "dataset.pth".

    Returns:
        Dataset: Generated dataset
    """
    if not os.path.exists(saved_path):
        regenerate = True  # if no found, then generate anyway
    if regenerate:
        meta_df = load_meta_data()
        # NOTE: Only focusing on dinner and lunch
        # meta_df = meta_df[meta_df["meal_type"] == "Lunch/Dinner"]
        train_df, val_df, test_df = generate_split(meta_df)
        # Create dataset
        # NOTE: Start counting
        train_dataset = iterate_dataframe(train_df)
        val_dataset = iterate_dataframe(val_df)
        test_dataset = iterate_dataframe(test_df)
        torch.save((train_dataset, val_dataset, test_dataset), saved_path)
        print(f"Leftover Dataset saved to {saved_path}!")
        return train_dataset, val_dataset, test_dataset
    return torch.load(saved_path)  # otherwise return pre-saved


def generate_split(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    # First split to get train and temp (val + test)
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df["leftover"]
    )
    # Calculate the ratio of val and test in the temp set
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    # Second split to get val and test from temp
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_test_ratio), stratify=temp_df["leftover"]
    )
    return train_df, val_df, test_df


# Define your custom dataset
class LeftoverDataset(Dataset):
    def __init__(self, data_len: int):
        self.before_img = [None] * data_len
        self.after_img = [None] * data_len
        self.leftover_label = [None] * data_len
        self.curr_idx = 0
        self.data_len = data_len

    def append(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        leftover_label: str,
    ):
        self.before_img[self.curr_idx] = before_img
        self.after_img[self.curr_idx] = after_img
        self.leftover_label[self.curr_idx] = leftover_label
        self.curr_idx += 1

    def convert_to_torch(self):
        # Converting to torch
        self.before_img = torch.from_numpy(np.array(self.before_img)).float()
        self.after_img = torch.from_numpy(np.array(self.after_img)).float()
        self.leftover_label = torch.from_numpy(np.array(self.leftover_label)).long()
        # Transposing to put channel before width and height
        self.before_img = self.before_img.permute(0, 1, 4, 2, 3).contiguous()
        self.after_img = self.after_img.permute(0, 1, 4, 2, 3).contiguous()
        self.targets = self.leftover_label

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # idx, task_id = idx
        before_img = self.before_img[idx]
        after_img = self.after_img[idx]
        target = self.leftover_label[idx]
        data = (before_img, after_img)
        return data, target


def load_meta_data(
    json_dir="../meta_data_dir/",
) -> pd.DataFrame:
    """
    load_meta_data loading ALL meta data from the JSON file

    Args:
        json_dir (str, optional): _description_. Defaults to "meta_data_dir/".

    Returns:
        pd.DataFrame: _description_
    """
    df_rows = []
    for subject_id in range(1, 50):  # loading all 49 subjects
        subject_id_str = f"{subject_id:03d}"
        json_name = json_dir + f"label_output_CaM01-{subject_id_str}.json"
        if not os.path.exists(json_name):  # skip non-existing files
            continue
        with open(json_name) as f:
            subject_meals = json.load(f)
        # NOTE: decoupling the JSON from a dictionary into a dataframe
        for meal_key in subject_meals.keys():
            if "meal" != meal_key[:4]:  # skip non-meal keys
                continue
            meal_data = subject_meals[meal_key]
            meal_data["subject_id"] = subject_id
            meal_data["meal_id"] = meal_key[4:]  # remove 'meal' prefix
            df_rows.append(meal_data)
    df = pd.DataFrame(df_rows)
    return df


def get_img(
    img_name: str,
    # img_dir="/home/data/datasets/CGM_Leftover_Images/",
    img_dir="/scratch/CGM_Leftover_Images/",
    target_size=(crop_size, crop_size),
) -> np.ndarray:
    """
    get_img loads an image file and returns it as a numpy array
    Args:
        img_name (str): The name of the image file
        img_dir (str, optional): The directory where the image file is located. Defaults to "/home/data/datasets/CGM_Leftover_Images/".
        target_size (tuple, optional): desired image size after reshaping. Defaults to (crop_size, crop_size).

    Returns:
        np.ndarray: The image as a numpy array
    """
    # Loading names out
    sub_name, img_path = img_name[:-4].split("/")
    orig_img_path = img_dir + "img_dir/" + img_name
    depth_img_path = img_dir + "CaM_Output/depths/" + sub_name
    depth_img_path += "/" + img_path + "_depth.png"
    seg_img_path = img_dir + "CaM_Output/segmentations/" + sub_name
    seg_img_path += "/img/" + img_path + "_segmented.png"
    orig_img = cv2.resize(
        cv2.cvtColor(cv2.imread(orig_img_path), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )
    dep_img = cv2.resize(
        cv2.cvtColor(cv2.imread(depth_img_path), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )
    seg_img = cv2.resize(
        cv2.cvtColor(cv2.imread(seg_img_path), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )

    return orig_img, dep_img, seg_img


def leftover_label_convert(leftover_str: str):
    leftover_dict = {
        "No Leftover": 0,
        "Little Leftover": 1,
        "Some Leftover": 2,
        "Full": 3,
    }
    return leftover_dict[leftover_str]


if __name__ == "__main__":
    set_random_seed()
    # NOTE: Running this script to generate dataset, toggle regenerate to False if you want to only inspect the dataset
    train_set, val_set, test_set = get_dataset(regenerate=True)
    # NOTE: Create DataLoader based on the saved dataset
    train_loader = DataLoader(
        train_set,
        batch_size=9999,
        shuffle=False,
    )
    # NOTE: Iterate through the DataLoader
    for data, leftover_label in train_loader:
        before_img, after_img = data
        print(before_img.shape, after_img.shape, leftover_label.shape)
    # NOTE: repeating the same inspection on val and test set
    val_dataloader = DataLoader(
        val_set,
        batch_size=9999,
        shuffle=False,
    )
    for data, leftover_label in val_dataloader:
        before_img, after_img = data
        print(before_img.shape, after_img.shape, leftover_label.shape)
    test_dataloader = DataLoader(
        test_set,
        batch_size=9999,
        shuffle=False,
    )
    for data, leftover_label in test_dataloader:
        before_img, after_img = data
        print(before_img.shape, after_img.shape, leftover_label.shape)
    print("Completed generating dataset")
