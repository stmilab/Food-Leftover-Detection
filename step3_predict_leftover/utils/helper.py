import numpy as np
import torch


class Sicong_Norm:
    def __init__(self, arr=None, min_val: float = 0, max_val: float = 200):
        if arr is None:
            self.min = min_val
            self.max = max_val
        else:
            # Making sure input is numpy
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            self.min = arr.min()
            self.max = arr.max()

    def normalize(self, arr):
        return (arr - self.min) / (self.max - self.min)

    def denormalize(self, arr):
        return arr * (self.max - self.min) + self.min

    norm = normalize
    denorm = denormalize
