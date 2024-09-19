import gzip
import json
import os.path as osp
import random
import time

import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset
from torchvision import transforms
from ray_diffusion.dataset.distortion_pattern.augment_distortion import augment_distortion, generatepindata, _crop_flow
from ray_diffusion.dataset.distortion_pattern.distortion_model import distortionModel, distortionParameter
from ray_diffusion.dataset.distortion_pattern.fisheye import RandomFisheye
from kornia.augmentation import AugmentationSequential
import kornia 
# import gryds

CO3D_DATA_DIR = "/ocean/projects/cis240055p/liuyuex/gemma/datasets/co3d_data"  # update this
CO3D_ANNOTATION_DIR = osp.join(CO3D_DATA_DIR, "co3d_annotations")
CO3D_DIR = osp.join(CO3D_DATA_DIR, "co3d")
CO3D_ORDER_PATH = osp.join(
    CO3D_DATA_DIR, "co3d_v2_random_order_{sample_num}/{category}.json"
)

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


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


def construct_camera_from_batch(batch, device):
    if isinstance(device, int):
        device = f"cuda:{device}"

    return PerspectiveCameras(
        R=batch["R"].reshape(-1, 3, 3),
        T=batch["T"].reshape(-1, 3),
        focal_length=batch["focal_lengths"].reshape(-1, 2),
        principal_point=batch["principal_points"].reshape(-1, 2),
        image_size=batch["image_sizes"].reshape(-1, 2),
        device=device,
    )


def save_batch_images(images, fname):
    cmap = plt.get_cmap("hsv")
    num_frames = len(images)
    num_rows = len(images)
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows):
        for j in range(4):
            if i < num_frames:
                axs[i * 4 + j].imshow(unnormalize_image(images[i][j]))
                for s in ["bottom", "top", "left", "right"]:
                    axs[i * 4 + j].spines[s].set_color(cmap(i / (num_frames)))
                    axs[i * 4 + j].spines[s].set_linewidth(5)
                axs[i * 4 + j].set_xticks([])
                axs[i * 4 + j].set_yticks([])
            else:
                axs[i * 4 + j].axis("off")
    plt.tight_layout()
    plt.savefig(fname)


def jitter_bbox(square_bbox, jitter_scale=(1.1, 1.2), jitter_trans=(-0.07, 0.07)):
    square_bbox = np.array(square_bbox.astype(float))
    s = np.random.uniform(jitter_scale[0], jitter_scale[1])
    tx, ty = np.random.uniform(jitter_trans[0], jitter_trans[1], size=2)
    side_length = square_bbox[2] - square_bbox[0]
    center = (square_bbox[:2] + square_bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s
    ul = center - extent
    lr = ul + 2 * extent
    return np.concatenate((ul, lr))


class Co3dDataset(Dataset):
    def __init__(
        self,
        category=("all_train",),
        split="train",
        transform=None,
        num_images=2,
        img_size=448,
        mask_images=False,
        crop_images=False,
        co3d_dir=None,
        co3d_annotation_dir=None,
        apply_augmentation=False,
        normalize_cameras=True,
        no_images=False,
        sample_num=None,
        seed=0,
        load_extra_cameras=False,
    ):
        start_time = time.time()

        self.category = category
        self.split = split
        self.transform = transform
        self.num_images = num_images
        self.img_size = img_size
        self.mask_images = mask_images
        self.crop_images = crop_images
        self.apply_augmentation = apply_augmentation
        self.normalize_cameras = normalize_cameras
        self.no_images = no_images
        self.sample_num = sample_num
        self.load_extra_cameras = load_extra_cameras

        if self.apply_augmentation:
            self.jitter_scale = (1.1, 1.2)
            self.jitter_trans = (-0.07, 0.07)
        else:
            # Note if trained with apply_augmentation, we should still use
            # apply_augmentation at test time.
            self.jitter_scale = (1, 1)
            self.jitter_trans = (0.0, 0.0)

        if co3d_dir is not None:
            self.co3d_dir = co3d_dir
            self.co3d_annotation_dir = co3d_annotation_dir
        else:
            self.co3d_dir = CO3D_DIR
            self.co3d_annotation_dir = CO3D_ANNOTATION_DIR

        if isinstance(self.category, str):
            self.category = [self.category]

        if "all_train" in self.category:
            self.category = TRAINING_CATEGORIES
        if "all_test" in self.category:
            self.category = TEST_CATEGORIES
        if "full" in self.category:
            self.category = TRAINING_CATEGORIES + TEST_CATEGORIES
        self.category = sorted(self.category)
        self.is_single_category = len(self.category) == 1

        # Fixing seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        for c in self.category:
            annotation_file = osp.join(
                self.co3d_annotation_dir, f"{c}_{self.split}.jgz"
            )
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < self.num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous and rotations are valid
                    det = np.linalg.det(data["R"])
                    if (np.abs(data["T"]) > 1e5).any() or det < 0.99 or det > 1.01:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

            print(f"Loaded {counter} instances of the {c} category.")

        self.sequence_list = list(self.rotations.keys())

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")
        print(f"Data loading took {(time.time()-start_time)} seconds.")

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, index):
        num_to_load = self.num_images if not self.load_extra_cameras else 8

        sequence_name = self.sequence_list[index % len(self.sequence_list)]
        metadata = self.rotations[sequence_name]

        if self.sample_num is not None:
            with open(
                CO3D_ORDER_PATH.format(
                    sample_num=self.sample_num, category=self.category[0]
                )
            ) as f:
                order = json.load(f)
            ids = order[sequence_name][:num_to_load]
        else:
            ids = np.random.choice(len(metadata), num_to_load, replace=False)

        return self.get_data(index=index, ids=ids)

    def _get_scene_scale(self, sequence_name):
        n = len(self.rotations[sequence_name])

        R = torch.zeros(n, 3, 3)
        T = torch.zeros(n, 3)

        for i, ann in enumerate(self.rotations[sequence_name]):
            R[i, ...] = torch.tensor(self.rotations[sequence_name][i]["R"])
            T[i, ...] = torch.tensor(self.rotations[sequence_name][i]["T"])

        cameras = PerspectiveCameras(R=R, T=T)
        cc = cameras.get_camera_center()
        centeroid = torch.mean(cc, dim=0)
        diff = cc - centeroid

        norm = torch.norm(diff, dim=1)
        scale = torch.max(norm).item()

        return scale

    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def _transform_intrinsic(self, image, bbox, principal_point, focal_length):
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

        return new_principal_ndc.astype(np.float32), new_focal_ndc.astype(np.float32)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        if sequence_name is None:
            index = index % len(self.sequence_list)
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        # Read image & camera information from annotations
        annos = [metadata[i] for i in ids]
        images = []
        images_d = []
        flows = []
        image_sizes = []
        PP = []
        FL = []
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

        for anno in annos:
            filepath = anno["filepath"]

            if not no_images:
                image = Image.open(osp.join(self.co3d_dir, filepath)).convert("RGB")

                # Optionally mask images with black background
                if self.mask_images:
                    black_image = Image.new("RGB", image.size, (0, 0, 0))
                    mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                    mask_path = osp.join(
                        self.co3d_dir, category, sequence_name, "masks", mask_name
                    )
                    mask = Image.open(mask_path).convert("L")

                    if mask.size != image.size:
                        mask = mask.resize(image.size)
                    mask = Image.fromarray(np.array(mask) > 125)
                    image = Image.composite(image, black_image, mask)

                # Determine crop, Resnet wants square images
                bbox_init = (
                    anno["bbox"]
                    if self.crop_images
                    else [0, 0, image.width, image.height]
                )
                bbox = square_bbox(np.array(bbox_init))
                if self.apply_augmentation:
                    bbox = jitter_bbox(
                        bbox,
                        jitter_scale=self.jitter_scale,
                        jitter_trans=self.jitter_trans,
                    )
                bbox = np.around(bbox).astype(int)

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

                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

                # Crop and normalize image
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
                ######
                # DEBUG visualizations
                # torchvision.utils.save_image(image[:, :, :], "image.png")
                # torchvision.utils.save_image(image_distort[:, :, :], "image_d.png")
                
                ######
                # _, canny_mask = kornia.filters.canny(image_distort.unsqueeze(0))
                images_d.append(image_distort[:, : self.img_size, : self.img_size])
                images.append(image[:, : self.img_size, : self.img_size])
                flows.append(flow[:, : self.img_size, : self.img_size])
                crop_parameters.append(crop_params)
            else:
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

            PP.append(principal_point)
            FL.append(focal_length)
            image_sizes.append(torch.tensor([self.img_size, self.img_size]))
            filenames.append(filepath)

        if not no_images:
            if self.load_extra_cameras:
                # Remove the extra loaded image, for saving space
                images = images[: self.num_images]
                images_d = images_d[: self.num_images]
                flows = flows[: self.num_images]
                # cannys = cannys[: self.num_images]

            images = torch.stack(images)
            images_d = torch.stack(images_d)
            flows = torch.stack(flows)
            crop_parameters = torch.stack(crop_parameters)
        else:
            images = None
            images_d = None
            flows = None
            crop_parameters = None

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])
        focal_lengths = torch.stack(FL)
        principal_points = torch.stack(PP)
        image_sizes = torch.stack(image_sizes)
        
        batch = {
            "model_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": torch.tensor(ids),
            "image": images,
            "image_d": images_d,
            "flow": flows,
            "R": R,
            "T": T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
        }

        return batch
