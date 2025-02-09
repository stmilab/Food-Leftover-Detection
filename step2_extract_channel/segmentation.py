import itertools
import logging
import math
import os
import sys
import urllib
import warnings
from functools import partial

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor

import dinov2.eval.segmentation.models
import dinov2.eval.segmentation.utils.colormaps as colormaps
import dinov2.eval.segmentation_m2f.models.segmentors

# Constants for model configuration
BACKBONE_SIZE = "small"  # Specify the backbone size: "small", "base", "large", or "giant"
HEAD_SCALE_COUNT = 3  # Number of scales for multi-scale head (1-5, higher is slower but better)
HEAD_DATASET = "voc2012"  # Dataset for segmentation: "ade20k" or "voc2012"
HEAD_TYPE = "ms"  # Type of segmentation head: "ms" (multi-scale) or "linear"
DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"  # Base URL for model and config downloads

# Define dataset colormaps and class names
DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}
DATASET_CLASS_NAMES = {
    "ade20k": colormaps.ADE20K_CLASS_NAMES,
    "voc2012": colormaps.VOC2012_CLASS_NAMES,
}

# URLs for configuration and checkpoint files
CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

# Custom module for center padding tensors to ensure compatibility with the backbone model
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple  # Padding to the nearest multiple of this value

    def _get_pad(self, size):
        # Calculate padding size for even distribution
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        # Apply padding to the input tensor
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

# Function to create a segmenter model with the specified configuration and backbone
def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)  # Initialize the segmentation model
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()  # Initialize model weights
    return model

# Load configuration file from a URL
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

# Load an image from a file and convert to RGB
def load_image_from_file(image_path: str) -> Image:
    return Image.open(image_path).convert("RGB")

# Render segmentation results into an image with class legends
def render_segmentation(segmentation_tensor, dataset):
    if dataset not in DATASET_COLORMAPS:
        raise ValueError(f"Dataset '{dataset}' is not found in DATASET_COLORMAPS.")
    colormap = DATASET_COLORMAPS[dataset]
    class_names = DATASET_CLASS_NAMES.get(dataset, [])
    colormap_array = np.array(colormap, dtype=np.uint8)

    # Convert tensor to NumPy array
    segmentation_logits = segmentation_tensor.cpu().numpy()

    if segmentation_logits.min() < -1 or segmentation_logits.max() >= len(colormap):
        raise ValueError("segmentation_logits contain values out of colormap range.")

    segmentation_values = colormap_array[segmentation_logits + 1]
    reshaped_values = segmentation_values.reshape(-1, 3)
    unique_colors = np.unique(reshaped_values, axis=0)

    color_to_class = {tuple(color): i for i, color in enumerate(colormap)}
    class_legend = {
        color_to_class[tuple(color)]: {"name": class_names[color_to_class[tuple(color)]], "color": tuple(color)}
        for color in unique_colors if tuple(color) in color_to_class
    }

    segmentation_image = Image.fromarray(segmentation_values)
    return segmentation_image, class_legend

def render_segmentation_legend(segmented_image, class_legend, image_path) -> None:
    # Display the segmented image with the legend
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(segmented_image)
    ax.axis('off')

    # Create the legend with normalized colors and class names
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(info['color']) / 255.0, markersize=10)
        for info in class_legend.values()
    ]
    labels = [info['name'] for info in class_legend.values()]

    # Add legend to plot
    ax.legend(handles, labels, loc='upper right', title="Classes", bbox_to_anchor=(1.2, 1))
    plt.savefig(image_path)

# Main function to segment an image and save the result
def segment_image(image_path: str, device: str = "cuda:7") -> torch.Tensor:
    """
    Segment an image using the pre-trained segmentation model.

    Args:
        image_path (str): The file path to the input image.
        device (str, optional): The device to run the model on. Defaults to "cuda:7".

    Returns:
        Image: The segmented image.
    """
    
    # Define backbone architecture
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    # Load backbone model
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()  # Set to evaluation mode
    backbone_model.to(device)

    # Load segmentation head configuration
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]

    # Create segmentation model
    model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.to(device)
    model.eval()

    # Load and preprocess the input image
    image = load_image_from_file(image_path)

    # Load segmentation configuration
    cfg_str = load_config_from_url(CONFIG_URL)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    # Create Mask2Former segmentation model
    model = init_segmentor(cfg)
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
    model.to(device)
    model.eval()

    # Perform segmentation inference
    array = np.array(image)[:, :, ::-1] # BGR
    segmentation_map = inference_segmentor(model, array)[0]

    # Dimensions: [H, W]
    return torch.from_numpy(segmentation_map).to(device)

# Example usage
segmentation_map = segment_image(
    image_path='/home/grads/n/nikhilnehra/projects/dinov2/images/CaM01-001/2021-09-19 12-00-16.jpg',
)

# Render and save the segmented image
segmented_image, class_legend = render_segmentation(segmentation_map, "ade20k")

output_path='utils/images/segmented_image.png'
segmented_image.save(output_path)
print("Segmented image saved to:", output_path)

render_segmentation_legend(segmented_image, class_legend, 'utils/images/segmented_image_legend.png')