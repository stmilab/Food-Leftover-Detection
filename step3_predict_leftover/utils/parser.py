import argparse
import sys
import os
import pdb

sys.path.append("utils/")
sys.path.append("models/")


def str2bool(v: str) -> bool:
    """
    str2bool
        Convert String into Boolean

    Arguments:
        v {str} --input raw string of boolean command

    Returns:
        bool -- converted boolean value
    """
    return v.lower() in ("yes", "true", "y", "t", "1")


def str2list(v: str) -> list:
    """
    str2list Convert String into sorted list

    Args:
        v (str): input raw string

    Returns:
        list: converted sorted list
    """
    if v == "":
        return []

    return list(sorted(v.split(",")))


def str2list_int(v: str) -> list:
    """
    str2list Convert String into sorted list of integers

    Args:
        v (str): input raw string

    Returns:
        list: converted sorted list of integers
    """
    if v == "":
        return []

    return sorted(int(item) for item in v.split(","))


def leftover_parser() -> argparse.Namespace:
    """
    sicong_argparse
        parsing command line arguments with reinforced formats

    Arguments:
        model {str} -- indicates which model being used

    Raises:
        RuntimeError: When the required model is unknown

    Returns:
        argparse.Namespace -- _description_
    """
    try:
        parser = argparse.ArgumentParser(description="Parser for Leftover Prediction")
        # NOTE: Hyperparameters
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Choosing the Batch Size, default=256 (If out of memory, try a smaller batch_size)",
        )
        parser.add_argument(
            "--img_model",
            default="ViT",
            help="Choosing the image model to be used, default=ViT (vision transformer)",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=128,
            help="Choosing the hidden size, default=128",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0,
            help="Weight Decay hyperparameter",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Choose the max number of epochs, default=3 for testing purpose",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="Define Learning Rate, default=1e-4, if failed try something smaller",
        )
        parser.add_argument(
            "--loss_type",
            default="CrossEntropy",
            help="Choosing the Loss function, default=AUC (or CrossEntropy)",
        )
        parser.add_argument(
            "--cen_threshold",
            type=float,
            default=0.2,
            help="Choosing the threshold for CEN, default=0.2",
        )
        # NOTE: Depedency based on local environment
        parser.add_argument(
            "--json_dir",
            default="meta_data_dir/",
            help="Please provide the Path to the JSON directory that contains meta data",
        )
        parser.add_argument(
            "--saved_path",
            default="utils/dataset.pth",
            help="Please provide the Path to the pre-saved dataset",
        )
        parser.add_argument(
            "--sel_gpu",
            type=str2list_int,
            default="3",
            help="Choosing which GPUs to use (STMI has GPU 0~7)",
        )
        parser.add_argument(
            "--shuffle_data",
            type=str2bool,
            default=True,
            help="Whether to shuffle data before train/test split",
        )
        parser.add_argument(
            "--training_size",
            type=float,
            default=0.8,
            help="Define how much portion of data is trained (default 0.8)",
        )
        # NOTE: Weights & Bias Parameters
        parser.add_argument(
            "--use_wandb",
            type=str2bool,
            default=False,
            help="Whether to save progress and results to Wandb",
        )
        parser.add_argument(
            "--wandb_tag",
            type=str,
            default="CEN",
            help="If using Wandb, define tag to help filter results",
        )
        flags, _ = parser.parse_known_args()
        # setting cuda device
        if 0 > min(flags.sel_gpu):
            flags.device = "cpu"
        else:
            gpu_str = ",".join(str(gpu) for gpu in flags.sel_gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            flags.device = "cuda"
        print("Flags:")
        for k, v in sorted(vars(flags).items()):
            print("\t{}: {}".format(k, v))
        return flags
    except Exception as error_msg:
        print(error_msg)


if __name__ == "__main__":
    flags = leftover_parser()
    print("This main func is used for testing purpose only")
