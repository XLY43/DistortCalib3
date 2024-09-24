import os, io, glob, natsort, random
import h5py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from kornia.augmentation import AugmentationSequential
from ray_diffusion.dataset.distortion_pattern.fisheye import RandomFisheye
import kornia 

TEST_CATEGORIES = [
    "0000",
    "0020",
    "0100",
    "0200",
    "0240",
    "0402",
    "1001",
    "5000",
    "0860",
    "1589",
]

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

def _transform_intrinsic(image, bbox, principal_point, focal_length):
    # Rescale intrinsics to match bbox
    half_box = np.array([image.width, image.height]).astype(np.float32) / 2
    org_scale = min(half_box).astype(np.float32)

    # Pixel coordinates
    principal_point_px = half_box - (np.array(principal_point) * org_scale)
    focal_length_px = np.array(focal_length) * org_scale
    principal_point_px -= bbox[:2]
    new_bbox = (bbox[2:] - bbox[:2]) / 2
    new_scale = min(new_bbox)

    # NDC coordinates
    new_principal_ndc = (new_bbox - principal_point_px) / new_scale
    new_focal_ndc = focal_length_px / new_scale

    principal_point = torch.tensor(new_principal_ndc.astype(np.float32))
    focal_length = torch.tensor(new_focal_ndc.astype(np.float32))

    return principal_point, focal_length

class MegaDepth(Dataset):
    def __init__(self, data_root="/ocean/projects/cis240055p/liuyuex/gemma/datasets/wildcamera/MonoCalib/MegaDepth/intrinsics", 
            img_size=448,
            shuffleseed=None, 
            split='train') -> None:
        
        data_names = glob.glob(os.path.join(data_root, '*.npz'))

        if split == 'train':
            data_names = [x for x in data_names if not any([cat in x for cat in TEST_CATEGORIES])]
            if shuffleseed is not None:
                random.seed(shuffleseed)
            random.shuffle(data_names)
        else:
            data_names = [x for x in data_names if any([cat in x for cat in TEST_CATEGORIES])]

        data_names_clean = []
        for idx in range(len(data_names)-1):
            intrinsics_path = data_names[idx]
            intrinsics = dict(np.load(intrinsics_path, allow_pickle=True))['intrinsics'].item()
            if len(intrinsics.keys()) != 0:
                data_names_clean.append(intrinsics_path)

        self.data_root = data_root
        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size, antialias=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        self.data_names = data_names_clean
        self.img_size = img_size
        self.num_to_load = 8
        self.datasetname = 'MegaDepth'

    def __len__(self):
        return len(self.data_names)

    def load_im(self, im_ref):
        im = Image.open(im_ref)
        return im

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
        p0_pytorch3d = (-(principal_point - c0) / scale)
        rotation = world2camera[:3, :3]
        tvec = world2camera[:3, 3]
        R_pytorch3d = rotation.T
        T_pytorch3d = tvec
        R_pytorch3d[:, :2] *= -1
        T_pytorch3d[:2] *= -1
        return R_pytorch3d, T_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_shape

    def __getitem__(self, idx):
        intrinsics_path = self.data_names[idx]
        poses_path = intrinsics_path.replace('intrinsics', 'poses')
        rgb_path = intrinsics_path.replace('intrinsics', 'rgb')

        intrinsics = dict(np.load(intrinsics_path, allow_pickle=True))['intrinsics'].item()
        poses = dict(np.load(poses_path, allow_pickle=True))['extrinsics'].item()
        rgbs = dict(np.load(rgb_path, allow_pickle=True))['rgb'].item()

        ids = np.random.choice(len(intrinsics.keys()), self.num_to_load, replace=False)

        # Read image & camera information from annotations
        images = []
        images_d = []
        flows = []
        image_sizes = []
        PP = []
        FL = []
        R = []
        T = []
        crop_parameters = []
        filenames = []

        aug_type = random.choice(["barrel", "pincushion", "perspective"])

        # distort_grid = get_grid(aug_type)
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
        
        frames = [*intrinsics.keys()]
        for id in ids:
            frame_name = frames[id]
            K = torch.from_numpy(intrinsics[frame_name]).float()
            pose = torch.from_numpy(poses[frame_name]).float()
            image = self.load_im(rgbs[frame_name])
            
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

            R_pytorch3d, T_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_shape = self._opencv_camera2pytorch3d(pose, K, [width, height])
            p0_pytorch3d, focal_pytorch3d = _transform_intrinsic(image, bbox, p0_pytorch3d, focal_pytorch3d)

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

            images.append(image[:, : self.img_size, : self.img_size]) # [3, 448, 448]
            images_d.append(image_distort[:, : self.img_size, : self.img_size])
            flows.append(flow[:, : self.img_size, : self.img_size]) #[2, 448, 448]
            R.append(R_pytorch3d) #[3, 3]
            T.append(T_pytorch3d) #[3]
            PP.append(p0_pytorch3d.to(torch.float32))
            FL.append(focal_pytorch3d.to(torch.float32))
            image_sizes.append(torch.tensor([self.img_size, self.img_size]))
            crop_parameters.append(crop_params)

            filenames.append(frame_name)
            
        images = torch.stack(images)
        images_d = torch.stack(images_d)
        flows = torch.stack(flows)
        crop_parameters = torch.stack(crop_parameters)
        R = torch.stack(R)
        T = torch.stack(T)
        PP = torch.stack(PP)
        FL = torch.stack(FL)
        image_sizes = torch.stack(image_sizes)

        batch = {
            "model_id": self.data_names[idx],
            "category": self.data_names[idx].split('/')[-1][:-4],
            "n": len(frames),
            "ind": torch.tensor(ids),
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