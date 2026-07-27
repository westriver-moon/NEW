from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image


class ImageTransform:
    def __init__(
        self,
        size: Sequence[int],
        training: bool,
        require_input_size: bool = False,
        grayscale: bool = False,
    ):
        self.height, self.width = int(size[0]), int(size[1])
        self.training = bool(training)
        self.require_input_size = bool(require_input_size)
        self.grayscale = bool(grayscale)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.require_input_size and image.size != (self.width, self.height):
            raise ValueError(
                "input image size %s does not match required size %s"
                % (image.size, (self.width, self.height))
            )
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.BICUBIC)
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.grayscale:
            image = image.convert("L").convert("RGB")
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(array)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std
