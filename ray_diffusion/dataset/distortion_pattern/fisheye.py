
import torch
import numpy as np

import kornia
from kornia.augmentation import AugmentationSequential

from typing import Any, Dict, Optional

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.core import Tensor
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid


def get_grid(distortion_type):
    if distortion_type not in ["barrel", "pincushion", "perspective", "random", "none", "shear"]:
        distortion_type = "random"

    if distortion_type == "barrel":
        coeff = np.sort(np.random.rand(5))
        grid = torch.tensor([[[ coeff[2], coeff[3], coeff[4],  coeff[3], coeff[2]],
                            [ coeff[1], coeff[2], coeff[2], coeff[2], coeff[1]],
                            [0,          0,  0,      0,  0],
                            [ -coeff[1], -coeff[1], -coeff[2],    -coeff[1], -coeff[3]],
                            [ -coeff[2], -coeff[3], -coeff[4],    -coeff[3], -coeff[2]]],

                            [[ coeff[2], coeff[1],  0,         -coeff[1],  -coeff[2]],
                            [ coeff[3], coeff[2],  0,         -coeff[2],  -coeff[3]],
                            [coeff[4],  coeff[2],  0,         -coeff[2],  -coeff[4]],
                            [coeff[3], coeff[2],   0,          -coeff[2], -coeff[3]],
                            [coeff[2],  coeff[1],  0,         -coeff[1], -coeff[2]]]])
        return grid.reshape([2, 5, 5])
    elif distortion_type == "pincushion":
        coeff = np.sort(np.random.rand(5))*0.2
        grid = torch.tensor([[[ coeff[2], coeff[3], coeff[4],  coeff[3], coeff[2]],
                            [ coeff[1], coeff[2], coeff[2], coeff[2], coeff[1]],
                            [0,          0,  0,      0,  0],
                            [ -coeff[1], -coeff[1], -coeff[2],    -coeff[1], -coeff[3]],
                            [ -coeff[2], -coeff[3], -coeff[4],    -coeff[3], -coeff[2]]],

                            [[ coeff[2], coeff[1],  0,         -coeff[1],  -coeff[2]],
                            [ coeff[3], coeff[2],  0,         -coeff[2],  -coeff[3]],
                            [coeff[4],  coeff[2],  0,         -coeff[2],  -coeff[4]],
                            [coeff[3], coeff[2],   0,          -coeff[2], -coeff[3]],
                            [coeff[2],  coeff[1],  0,         -coeff[1], -coeff[2]]]])
        return (-1)*grid.reshape([2, 5, 5])
    elif distortion_type == "perspective":
        coeff = np.sort(np.random.rand(5))*0.1
        grid = torch.tensor([[[ -coeff[4], -coeff[2], -coeff[0],  -coeff[2], -coeff[4]],
                       [ -coeff[3], -coeff[1], -coeff[0], -coeff[1], -coeff[3]],
                       [0,          0,  0,      0,  0],
                       [ coeff[3], coeff[1], coeff[0],    coeff[1], coeff[3]],
                       [ coeff[4], coeff[2], coeff[0],    coeff[2], coeff[4]]],

                      [[ -coeff[4], -coeff[3],  0,         coeff[3],  coeff[4]],
                       [ -coeff[2], -coeff[1],  0,         coeff[1],  coeff[2]],
                       [-coeff[0],  -coeff[0],  0,         coeff[0],  coeff[0]],
                       [-coeff[2], -coeff[1],   0,          coeff[1], coeff[2]],
                       [-coeff[4],  -coeff[3],  0,          coeff[3], coeff[4]]]])
        return grid.reshape([2, 5, 5])

    elif distortion_type == "random":
        random_grid = np.random.rand(2, 5, 5)
        random_grid -= 0.5
        random_grid /= 10
        return torch.tensor(random_grid)

    elif distortion_type == "none":
        return torch.zeros(2, 5, 5)
    
    elif distortion_type == "shear":
        factor = (np.random.rand(1)-0.5)*0.4
        coeff = [0., factor[0]/5*1, factor[0]/5*2, factor[0]/5*3, factor[0]/5*4]
        grid = torch.tensor([[[0,          0,  0,      0,  0],
                            [0,          0,  0,      0,  0],
                            [0,          0,  0,      0,  0],
                            [0,          0,  0,      0,  0],
                            [0,          0,  0,      0,  0]],
                            [[ coeff[4], coeff[4],  coeff[4],         coeff[4],  coeff[4]],
                            [ coeff[3], coeff[3],  coeff[3],         coeff[3],  coeff[3]],
                            [ coeff[2], coeff[2],  coeff[2],         coeff[2],  coeff[2]],
                            [ coeff[1], coeff[1],  coeff[1],         coeff[1],  coeff[1]],
                            [ 0, 0, 0, 0, 0]]])
        return grid.reshape([2, 5, 5])

    

class RandomFisheye(AugmentationBase2D):
    r"""Add random camera radial distortion.

    .. image:: _static/img/RandomFisheye.png

    Args:
        center_x: Ranges to sample respect to x-coordinate center with shape (2,).
        center_y: Ranges to sample respect to y-coordinate center with shape (2,).
        gamma: Ranges to sample for the gamma values respect to optical center with shape (2,).
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Examples:
        >>> import torch
        >>> img = torch.ones(1, 1, 2, 2)
        >>> center_x = torch.tensor([-.3, .3])
        >>> center_y = torch.tensor([-.3, .3])
        >>> gamma = torch.tensor([.9, 1.])
        >>> out = RandomFisheye(center_x, center_y, gamma)(img)
        >>> out.shape
        torch.Size([1, 1, 2, 2])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomFisheye(center_x, center_y, gamma, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        center_x: Tensor,
        center_y: Tensor,
        gamma: Tensor,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._check_tensor(center_x)
        self._check_tensor(center_y)
        self._check_tensor(gamma)
        self._param_generator = rg.PlainUniformGenerator(
            (center_x[:, None], "center_x", None, None),
            (center_y[:, None], "center_y", None, None),
            (gamma[:, None], "gamma", None, None),
        )

    def _check_tensor(self, data: Tensor) -> None:
        if not isinstance(data, Tensor):
            raise TypeError(f"Invalid input type. Expected Tensor - got: {type(data)}")

        if len(data.shape) != 1 and data.shape[0] != 2:
            raise ValueError(f"Tensor must be of shape (2,). Got: {data.shape}.")

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # create the initial sampling fields
        B, _, H, W = input.shape
        grid = create_meshgrid(H, W, normalized_coordinates=True)
        field_x = grid[..., 0].to(input)  # 1xHxW
        field_y = grid[..., 1].to(input)  # 1xHxW
        # vectorize the random parameters
        center_x = params["center_x"].view(B, 1, 1).to(input)
        center_y = params["center_y"].view(B, 1, 1).to(input)
        gamma = params["gamma"].view(B, 1, 1).to(input)
        # compute and apply the distances respect to the camera optical center
        distance = ((center_x - field_x) ** 2 + (center_y - field_y) ** 2) ** 0.5
        field_x = (field_x + field_x * distance**gamma)*0.5  # BxHxw
        field_y = (field_y + field_y * distance**gamma)*0.5  # BxHxW
        return remap(input, field_x, field_y, normalized_coordinates=True, align_corners=False), field_x, field_y