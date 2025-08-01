import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

from exchange import ModuleParallel, LayerNormParallel, conv1x1, Exchange


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                        as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W`, C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H`*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


# TODO: FINISH implementing


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, threshold=0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                        (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.attn = ModuleParallel(
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        )

        self.layer_norm_2 = LayerNormParallel(embed_dim, num_parallel=3)
        self.linear = nn.Sequential(
            conv1x1(embed_dim, hidden_dim),
            ModuleParallel(nn.GELU(inplace=True)),
            ModuleParallel(nn.Dropout(dropout)),
            conv1x1(hidden_dim, embed_dim),
            ModuleParallel(nn.Dropout(dropout)),
        )
        # Implementing exchange
        self.exchange = Exchange()
        self.ln_list = []
        for module in self.layer_norm_2.modules():
            if isinstance(module, nn.LayerNorm):
                self.ln_list.append(module)

    def forward(self, x, threshold=0):
        inp_x = self.layer_norm_1(x)
        attn_output = self.attn(inp_x, inp_x, inp_x)
        x = x + attn_output
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1 + num_patches, embed_dim))

    def forward(self, x, return_attn=False):
        # Preprocess input
        x = img_to_patch(
            x, self.patch_size
        )  # [(), num_channels * patch_size * patch_size]
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[: T + 1, :]  # new
        # x = x + self.pos_embedding[:, : T + 1] # old

        attn_maps = []
        for transformer in self.transformer:
            if return_attn:
                x, attn_map = transformer(x, return_attn)
                attn_maps.append(attn_map)
            else:
                x = transformer(x)

        x = self.dropout(x)
        x = x.transpose(0, 1)
        cls = x[0]
        out = self.mlp_head(cls)
        if return_attn:
            return out, attn_maps
        return out


class ViT(nn.Module):
    """
    ViT Vision Transformer for image processing and training

    _extended_summary_

    Args:
        nn (torch.nn): This is a torch nueural network class
    """

    def __init__(self, model_kwargs):
        super().__init__()
        self.model = VisionTransformer(
            **model_kwargs
        )  # expects dictionary {key: value} pairs
        # self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
