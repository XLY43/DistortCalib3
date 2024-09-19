#!/usr/bin/env python3

import os
import re
import bz2
import glob
import torch
import pickle
import numpy as np
import random

import kornia
from os import path
from PIL import Image
from copy import copy
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import functional as F

from ray_diffusion.dataset.distortion_pattern.augment_distortion import augment_distortion, generatepindata, _crop_flow
from ray_diffusion.dataset.distortion_pattern.distortion_model import distortionModel, distortionParameter
from ray_diffusion.dataset.distortion_pattern.fisheye import RandomFisheye
from kornia.augmentation import AugmentationSequential


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.
    Args:
        bbox: Bounding box in xyxy format (4,).
    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox


class TartanAir(Dataset):
    def __init__(self, 
                root="/ocean/projects/cis240055p/liuyuex/gemma/datasets/tartanair", 
                catalog_path="/ocean/projects/cis240055p/liuyuex/gemma/datasets/tartanair/.cache/tartanair-sequences.pbz2", 
                scale=1, 
                img_size=448, 
                augment=True, 
                exclude=None, 
                include=None):
        super().__init__()
        self.img_size = img_size
        # self.augment = AirAugment(scale, size=[480, 640], resize_only=not augment)
        if catalog_path is not None and os.path.exists(catalog_path):
            with bz2.BZ2File(catalog_path, 'rb') as f:
                self.sequences, self.image, self.poses, self.sizes = pickle.load(f)
        else:
            self.sequences = glob.glob(os.path.join(root,'*','[EH]a[sr][yd]','*'))
            self.image, self.poses, self.sizes = {}, {}, []
            ned2den = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            for seq in self.sequences:
                quaternion = np.loadtxt(path.join(seq, 'pose_left.txt'), dtype=np.float32)
                self.poses[seq] = ned2den @ pose2mat(quaternion)
                self.image[seq] = sorted(glob.glob(path.join(seq,'image_left','*.png')))
                assert(len(self.image[seq])==self.poses[seq].shape[0])
                self.sizes.append(len(self.image[seq]))
            # import pdb; pdb.set_trace()
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with bz2.BZ2File(catalog_path, 'wb') as f:
                pickle.dump((self.sequences, self.image, self.poses, self.sizes), f)
        # Camera Intrinsics of TartanAir Dataset
        fx, fy, cx, cy = img_size//2, img_size//2, img_size//2, img_size//2
        self.K = torch.FloatTensor([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        # include/exclude seq with regex
        incl_pattern = re.compile(include) if include is not None else None
        excl_pattern = re.compile(exclude) if exclude is not None else None
        final_list = []
        for seq, size in zip(self.sequences, self.sizes):
            if (incl_pattern and incl_pattern.search(seq) is None) or \
                    (excl_pattern and excl_pattern.search(seq) is not None):
                del self.poses[seq], self.image[seq]
            else:
                final_list.append((seq, size))
        self.sequences, self.sizes = zip(*final_list) if len(final_list) > 0 else ([], [])

        # simple base transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.img_size, antialias=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def _opencv_camera2pytorch3d(self, world2camera, camera_K, image_shape):
        """Convert OpenCV camera convension to pytorch3d convension

        Args:
            world2camera (ndarray): 4x4 matrix
            camera_K (ndarray): 3x3 intrinsic matrix
            image_shape (ndarray): [image_heigt, width]
        """
        focal_length = np.asarray([camera_K[0, 0], camera_K[1, 1]])
        principal_point = camera_K[:2, 2]
        image_size_wh = np.asarray([image_shape[1], image_shape[0]])
        scale = (image_size_wh.min() / 2.0)
        c0 = image_size_wh / 2.0

        # Get the PyTorch3D focal length and principal point.
        focal_pytorch3d = torch.tensor(focal_length / scale)
        p0_pytorch3d = torch.tensor((-(principal_point - c0) / scale).clone())
        rotation = world2camera[:3, :3]
        tvec = world2camera[:3, 3]
        R_pytorch3d = rotation.T
        T_pytorch3d = tvec
        R_pytorch3d[:, :2] *= -1
        T_pytorch3d[:2] *= -1
        return R_pytorch3d, T_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_shape

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, index):
        seq, K = self.sequences[index], self.K

        num_to_load = 8

        frames = np.random.choice(len(self.image[seq]), num_to_load, replace=False)

        images = []
        images_d = []
        flows = []
        poses = []
        PP = []
        FL = []
        R = []
        T = []
        image_sizes = []
        crop_parameters = []
        filenames = []

        # define augmentation
        aug_type = random.choice(["barrel", "pincushion", "perspective"])
        if aug_type == "barrel":
            center_x = torch.tensor([0.0, 0.0])
            center_y = torch.tensor([0.0, 0.0])
            gamma = torch.tensor([1.5, 0.1])
            aug_forms = RandomFisheye(center_x, center_y, gamma, p=1)
        elif aug_type == "pincushion":
            center_x = torch.tensor([0.0, 0.0])
            center_y = torch.tensor([0.0, 0.0])
            gamma = torch.tensor([-0.4, -0.1])
            aug_forms = RandomFisheye(center_x, center_y, gamma, p=1)
        elif aug_type == "perspective":
            aug_forms = AugmentationSequential(
                kornia.augmentation.RandomPerspective(0.2, p=1.0, keepdim=True),
                kornia.augmentation.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=["input", "keypoints"], 
                same_on_batch=True)
        elif aug_type == "none":
            aug_forms = AugmentationSequential(
                kornia.augmentation.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=["input", "keypoints"], 
                same_on_batch=True)

        for frame in frames:
            image = Image.open(self.image[seq][frame])

            bbox_init = [0, 0, image.width, image.height]
            bbox = square_bbox(np.array(bbox_init))

            # Crop parameters
            crop_center = (bbox[:2] + bbox[2:]) / 2
            # convert crop center to correspond to a "square" image
            width, height = image.size
            length = max(width, height)
            s = length / min(width, height)
            crop_center = crop_center + (length - np.array([width, height])) / 2
            # convert to NDC
            cc = s - 2 * s * crop_center / length
            crop_width = 2 * s * (bbox[2] - bbox[0]) / length
            crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])

            image = self._crop_image(image, bbox)
            image = self.transform(image)
            image = image[:, : self.img_size, : self.img_size]

            # Distort image
            if aug_type == "barrel" or aug_type == "pincushion":
                image_distort, field_x, field_y = aug_forms(image.unsqueeze(0))
                grid = kornia.utils.create_meshgrid(self.img_size, self.img_size, False).reshape(1, -1, 2) # batch size is 1
                warped_grid = torch.stack([field_x, field_y], dim=-1)
                grid = 2 * grid / (self.img_size - 1) - 1
                grid = grid.reshape(1, self.img_size, self.img_size, 2)
                flow = (grid - warped_grid).reshape(1, self.img_size, self.img_size, 2)
                flow[flow.abs() > 0.3] = 0
                image_distort = image_distort.squeeze(0)
                flow = flow.squeeze(0).permute(2, 0, 1)
            else:
                grid = kornia.utils.create_meshgrid(self.img_size, self.img_size, False).reshape(1, -1, 2) # batch size is 1
                out = aug_forms(image, grid)
                image_distort = out[0]
                warped_grid = out[1].reshape(1, -1, 2)
                grid = 2 * grid/ (self.img_size - 1) - 1
                warped_grid = 2 * warped_grid/ (self.img_size - 1) - 1
                flow = (warped_grid - grid).reshape(1, self.img_size, self.img_size, 2)
                image_distort = image_distort.squeeze(0)
                flow = flow.squeeze(0).permute(2, 0, 1)
            
            # import pdb; pdb.set_trace()
            # torchvision.utils.save_image(image[:, :, :], "image.png")
            # torchvision.utils.save_image(image_distort[:, :, :], "image_d.png")
            # flow_img = torchvision.utils.flow_to_image(flow)
            # torchvision.io.write_png(flow_img, "image_flow.png")

            pose = self.poses[seq][frame]
            
            R_pytorch3d, T_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_shape = self._opencv_camera2pytorch3d(pose, K, [self.img_size, self.img_size])
            images.append(image[:, : self.img_size, : self.img_size]) # [3, 448, 448]
            images_d.append(image_distort[:, : self.img_size, : self.img_size])
            flows.append(flow[:, : self.img_size, : self.img_size]) #[2, 448, 448]
            R.append(torch.tensor(pose[:3, :3].clone())) #[3, 3]
            T.append(torch.tensor(pose[:3, 3].clone())) #[3]
            PP.append(p0_pytorch3d.to(torch.float32))
            FL.append(focal_pytorch3d.to(torch.float32))
            image_sizes.append(torch.tensor([self.img_size, self.img_size]))
            crop_parameters.append(crop_params)

            filenames.append(self.image[seq][frame])

        images = torch.stack(images)
        images_d = torch.stack(images_d)
        flows = torch.stack(flows)
        R = torch.stack(R)
        T = torch.stack(T)
        PP = torch.stack(PP)
        FL = torch.stack(FL)
        image_sizes = torch.stack(image_sizes)
        crop_parameters = torch.stack(crop_parameters)
        
        batch = {
            "model_id": seq,
            "category": seq.split('/')[-3],
            "n": len(self.image[seq]),
            "ind": torch.tensor(frames),
            "image": images,
            "image_d": images_d,
            "flow": flows,
            "R": R,
            "T": T,
            "principal_point": PP,
            "focal_length": FL,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
        }
        return batch

    def rand_split(self, ratio, seed=42):
        total, ratio = len(self.sequences), np.array(ratio)
        split_idx = np.cumsum(np.round(ratio / sum(ratio) * total), dtype=np.int)[:-1]
        subsets = []
        for perm in np.split(np.random.default_rng(seed=seed).permutation(total), split_idx):
            subset = copy(self)
            subset.sequences = np.take(self.sequences, perm).tolist()
            subset.sizes = np.take(self.sizes, perm).tolist()
            subsets.append(subset)
        return subsets


class AirSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=True, overlap=True):
        self.data_sizes = data.sizes
        self.bs = batch_size
        self.shuffle = shuffle
        self.batches = []
        for i, size in enumerate(self.data_sizes):
            b_start = np.arange(0, size - self.bs, 1 if overlap else self.bs)
            self.batches += [list(zip([i]*self.bs, range(st, st+self.bs))) for st in b_start]
        if self.shuffle: np.random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def pose2mat(pose):
    """Converts pose vectors to matrices.
    Args:
      pose: [tx, ty, tz, qx, qy, qz, qw] (N, 7).
    Returns:
      [R t] (N, 3, 4).
    """
    t = pose[:, 0:3, None]
    rot = R.from_quat(pose[:, 3:7]).as_matrix().astype(np.float32).transpose(0, 2, 1)
    t = -rot @ t
    return torch.cat([torch.from_numpy(rot), torch.from_numpy(t)], dim=2)

