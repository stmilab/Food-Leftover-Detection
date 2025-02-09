import itertools
import math
import os
import sys
import urllib
from functools import partial

import matplotlib
import mmcv
import torch
import torch.nn.functional as F
from PIL import Image
from mmcv.runner import load_checkpoint
from torchvision import transforms

from dinov2.eval.depth.models import build_depther

# Configuration settings for backbone and head
BACKBONE_SIZE = "small" # Options: "small", "base", "large", "giant"
HEAD_DATASET = "nyu" # Dataset for head configuration: "nyu", "kitti"
HEAD_TYPE = "dpt" # Head type: "linear", "linear4", "dpt"

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2" # Base URL for downloading model configs and checkpoints

# Padding module to ensure image dimensions are multiples of a specific size
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        # Calculate padding sizes to achieve dimensions as multiples of 'multiple'
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        # Apply padding to tensor
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

# Create the depth estimation model
def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg") # Training configuration
    test_cfg = cfg.get("test_cfg") # Testing configuration
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    # Override the backbone's forward function to extract intermediate layers
    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    # Apply padding before the forward pass if the backbone model has a patch size
    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

# Load configuration file from a given URL
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

# Load an image from file and convert it to RGB
def load_image_from_file(image_path: str) -> Image:
    return Image.open(image_path).convert("RGB")

# Define a transform pipeline for depth estimation
def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53), # Normalize RGB channels
            std=(58.395, 57.12, 57.375),
        ),
    ])

# Render a depth image using a colormap
def render_depth(values, colormap_name="magma_r") -> Image:
    # Normalize values to range [0, 1]
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    # Apply a colormap and discard alpha channel
    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xHxWx4)
    colors = colors[:, :, :3] # Remove alpha component
    return Image.fromarray(colors)

# Main function to estimate depth and save the result
def estimate_depth(image_path: str, device: str = "cuda:7") -> torch.Tensor:
    """
    Estimate the depth of an input image using a pre-trained depth estimation model.

    Args:
        image_path (str): The file path to the input image.
        device (str, optional): The device to run the model on. Defaults to "cuda:7".

    Returns:
        Image: The estimated depth image.
    """
    # Define backbone architecture mappings
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    # Load pre-trained backbone model from Torch Hub
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval() # Set model to evaluation mode
    backbone_model.to(device) # Move model to specified device

    # URLs for configuration and checkpoint files
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    # Load model configuration
    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    # Create the depth estimation model
    model = create_depther(
        cfg,
        backbone_model=backbone_model,
        backbone_size=BACKBONE_SIZE,
        head_type=HEAD_TYPE,
    )

    # Load pre-trained checkpoint for the depth estimation model
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.eval() # Set model to evaluation mode
    model.to(device) # Move model to the specified device

    # Load and preprocess the input image
    image = load_image_from_file(image_path)
    transform = make_depth_transform()

    # Scale and transform the image
    scale_factor = 1
    rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).to(device) # Add batch dimension and move to device

    # Perform depth estimation
    with torch.inference_mode():
        depth_map = model.whole_inference(batch, img_meta=None, rescale=True)

    # Dimentions: [1, 1, H, W]
    return depth_map

# Example usage
depth_map = estimate_depth(
    image_path='/home/grads/n/nikhilnehra/projects/dinov2/images/CaM01-001/2021-09-19 12-00-16.jpg',
)

# Render and save the depth image
depth_image = render_depth(depth_map.squeeze().cpu())

output_path='utils/images/depth_image.png'
depth_image.save(output_path)
print("Depth image saved to:", output_path)