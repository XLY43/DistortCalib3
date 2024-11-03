# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional
import os 
import random 
from collections import defaultdict

import numpy as np

import torch

import webdataset as wds

from atek.data_loaders.atek_wds_dataloader import load_atek_wds_dataset
from atek.util.atek_constants import ATEK_CATEGORY_ID_TO_NAME, ATEK_CATEGORY_NAME_TO_ID
from atek.util.file_io_utils import load_yaml_and_extract_tar_list

from projectaria_tools.core.sophus import SE3

from webdataset.filters import pipelinefilter

from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from kornia.utils import create_meshgrid
import torchvision
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from torchvision import transforms


TRAIN_LIST = "/ocean/projects/cis240055p/liuyuex/gemma/ATEK/training_output/local_train_tars.yaml"
TEST_LIST = "/ocean/projects/cis240055p/liuyuex/gemma/ATEK/training_output/local_validation_tars.yaml"
CATEGORY_JSON = "/ocean/projects/cis240055p/liuyuex/gemma/ATEK/data/atek_id_to_name.json"


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


class CubeRCNNModelAdaptor:
    def __init__(
        self,
        # TODO: make these a DictConfig
        min_bb2d_area: Optional[float] = 100,
        min_bb3d_depth: Optional[float] = 0.3,
        max_bb3d_depth: Optional[float] = 5.0,
    ):
        self.min_bb2d_area = min_bb2d_area
        self.min_bb3d_depth = min_bb3d_depth
        self.max_bb3d_depth = max_bb3d_depth
        self.transform = transforms.Compose(
                [
                    # transforms.ToTensor(),
                    transforms.Resize(448, antialias=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            "mfcd#camera-rgb+images": "image",
            "mfcd#camera-rgb+projection_params": "camera_params",
            "mfcd#camera-rgb+camera_model_name": "camera_model",
            "mfcd#camera-rgb+t_device_camera": "t_device_rgbcam",
            "mfcd#camera-rgb+frame_ids": "frame_id",
            "mfcd#camera-rgb+capture_timestamps_ns": "timestamp_ns",
            "mtd#ts_world_device": "ts_world_device",
            "sequence_name": "sequence_name",
            "gt_data": "gt_data",
        }
        return dict_key_mapping

    def atek_to_cubercnn(self, data):
        """
        A helper data transform function to convert a ATEK webdataset data sample built by CubeRCNNSampleBuilder, to CubeRCNN unbatched
        samples. Yield one unbatched sample a time to use the collation and batching mechanism in
        the webdataset properly.
        """
        for atek_wds_sample in data:
            sample = {}
            self._update_camera_data_in_sample(atek_wds_sample, sample)
            self._update_T_world_camera(atek_wds_sample, sample)

            # # Skip if no gt data
            # if "gt_data" in atek_wds_sample and len(atek_wds_sample["gt_data"]) > 0:
            #     self._update_gt_data_in_sample(atek_wds_sample, sample)

            yield sample

    def _update_camera_data_in_sample(self, atek_wds_sample, sample):
        """
        Initialize sample image
        Process camera K-matrix information and update the sample dictionary.
        """
        # assert atek_wds_sample["image"].shape[0] == 1, "Only support 1 frame"
        image_height, image_width = atek_wds_sample["image"].shape[2:]

        # calculate K-matrix
        camera_model = atek_wds_sample["camera_model"]
        # assert (
        #     camera_model == "CameraModelType.LINEAR"
        # ), f"Only linear camera model supported in CubeRCNN model, this data has {camera_model} instead."
        k_matrix = torch.zeros((3, 3), dtype=torch.float32)
        params = atek_wds_sample["camera_params"]
        # k_matrix[0, 0], k_matrix[1, 1] = params[0], params[1]
        # k_matrix[0, 2], k_matrix[1, 2] = params[2], params[3]
        # k_matrix[2, 2] = 1.0

        focal_length = np.asarray([params[0]])
        principal_point = np.asarray([params[9], params[10]])

        image_size_wh = np.asarray([image_width, image_height])
        scale = (image_size_wh.min() / 2.0)
        c0 = image_size_wh / 2.0
        focal_pytorch3d = torch.tensor(focal_length / scale)
        p0_pytorch3d = torch.tensor((-(principal_point - c0) / scale))
        image_height, image_width = 448, 448

        # retrieve the flow from the sequence
        flows = []
        crop_parameters = []
        principal_points = []
        focal_lengths = []
        image_d = []
        image_sizes = []
        radial_params = []
        thinprism_params = []
        tangential_params = []
        for i in range(5):
            # import pdb; pdb.set_trace()
            # cur_img = F.interpolate(atek_wds_sample["image"][i].unsqueeze(0), [image_height, image_width], mode='bilinear', align_corners=True).squeeze().float()
            cur_img = atek_wds_sample["image"][i].float()/255.
            cur_img = self.transform(cur_img)
            # import pdb; pdb.set_trace()
            if len(cur_img.shape) != 3:
                cur_img = cur_img.unsqueeze(0)
            cam = FishEyeCameras(focal_length=torch.tensor([[focal_pytorch3d[0]]]).float(), 
                                principal_point=torch.tensor([[p0_pytorch3d[0], p0_pytorch3d[1]]]).float(), 
                                radial_params=torch.tensor([[params[3], params[4], params[5], params[6], params[7], params[8]]]).float(),
                                tangential_params=torch.tensor([[params[9], params[10]]]).float(),
                                thin_prism_params=torch.tensor([[params[11], params[12], params[13], params[14]]]).float())

            grid = create_meshgrid(image_height, image_width, normalized_coordinates=True)
            grid = grid.reshape(-1, 2)
            grid = torch.cat([grid, torch.ones(grid.shape[0], 1)], dim=1)
            # import pdb; pdb.set_trace()
            dist_grid = cam.unproject_points(grid)
            dist_grid = dist_grid[:, :2].reshape(1, image_height, image_width, 2)
            warped_grid = dist_grid
            grid = grid[:, :2].reshape(1, image_height, image_width, 2)
            flow = (grid - warped_grid).reshape(1, image_height, image_width, 2)#.clamp(-1,1)
            flow[flow.abs() > 0.3] = 0
            flows.append(flow.squeeze().permute(2, 0, 1))

            bbox_init = [0, 0, image_width, image_height]
            bbox = square_bbox(np.array(bbox_init))

            # Crop parameters
            crop_center = (bbox[:2] + bbox[2:]) / 2
            # convert crop center to correspond to a "square" image
            length = max(image_width, image_height)
            s = length / min(image_width, image_height)
            crop_center = crop_center + (length - np.array([image_width, image_height])) / 2
            # convert to NDC
            cc = s - 2 * s * crop_center / length
            crop_width = 2 * s * (bbox[2] - bbox[0]) / length
            crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])
            crop_parameters.append(crop_params)

            principal_points.append(p0_pytorch3d)
            focal_lengths.append(focal_pytorch3d)
            radial_params.append(torch.tensor([[params[3], params[4], params[5], params[6], params[7], params[8]]]).float())
            thinprism_params.append(torch.tensor([[params[11], params[12], params[13], params[14]]]).float())
            tangential_params.append(torch.tensor([[params[9], params[10]]]).float())
            image_d.append(cur_img.squeeze())
            image_sizes.append(torch.tensor([image_width, image_height]))

        PP = torch.stack(principal_points)
        FL = torch.stack(focal_lengths)
        radial_params = torch.stack(radial_params)
        thinprism_params = torch.stack(thinprism_params)
        tangential_params = torch.stack(tangential_params)
        flows = torch.stack(flows)
        crop_parameters = torch.stack(crop_parameters)
        image_d = torch.stack(image_d)
        image_sizes = torch.stack(image_sizes)
        sample.update( 
            {
                # rgb -> bgr
                # "image": atek_wds_sample["image"][:, [2, 1, 0], :, :].clone().detach(),
                "image_d": image_d,
                "image_size": image_sizes,
                # "K": k_matrix.tolist(),
                # "height": image_height,
                # "width": image_width,
                # "K_matrix": k_matrix,
                "timestamp_ns": atek_wds_sample["timestamp_ns"],
                "frame_id": atek_wds_sample["frame_id"],
                "sequence_name": atek_wds_sample["sequence_name"],
                # "camera_params": atek_wds_sample["camera_params"].tolist(),
                # "camera_model": atek_wds_sample["camera_model"],
                "principal_point": PP.float(),
                "focal_length": FL.float(),
                "radial_params": radial_params.float(),
                "thinprism_params": thinprism_params.float(),
                "tangential_params": tangential_params.float(),
                "flow": flows.float(),
                "crop_parameters": crop_parameters,
            }
        )

    def _update_T_world_camera(self, atek_wds_sample, sample):
        """
        Compute world-to-camera transformation matrices, and update this field in sample dict.
        """
        axis_conversion = torch.tensor([
                            [-1,  0,  0],
                            [ 0,  1,  0],
                            [ 0,  0, -1]
                        ], dtype=torch.float32)

        T_world_device = SE3.from_matrix3x4(atek_wds_sample["ts_world_device"]) #[0]
        # print(T_world_device)
        T_device_rgbCam = SE3.from_matrix3x4(atek_wds_sample["t_device_rgbcam"])
        T_world_rgbCam = T_world_device @ T_device_rgbCam
        T_world_camera = T_world_rgbCam.to_matrix3x4()
        T_world_camera = torch.tensor(T_world_camera, dtype=torch.float32)
        new_rotation = torch.matmul(axis_conversion, T_world_camera[..., :3])
        new_translation = torch.matmul(axis_conversion, T_world_camera[..., 3:])
        R = []
        T = []
        for i in range(5):
            new_translation[i] = new_translation[i] / new_translation[2]
            R.append(new_rotation[i])
            T.append(new_translation[i])
        sample["R"] = torch.stack(R).float()
        sample["T"] = torch.stack(T).squeeze().float()
        # sample["R"] = new_rotation
        # sample["T"] = new_translation

    def _process_2d_bbox_dict(self, bb2d_dict):
        """
        Process 2D bounding boxes by rearranging the bounding box coordinates to be
        in the order x0, y0, x1, y1 and calculating the area of each 2D bounding box.
        """
        bb2ds_x0y0x1y1 = bb2d_dict["box_ranges"]
        bb2ds_x0y0x1y1 = bb2ds_x0y0x1y1[:, [0, 2, 1, 3]]
        bb2ds_area = (bb2ds_x0y0x1y1[:, 2] - bb2ds_x0y0x1y1[:, 0]) * (
            bb2ds_x0y0x1y1[:, 3] - bb2ds_x0y0x1y1[:, 1]
        )

        return bb2ds_x0y0x1y1, bb2ds_area

    def _process_3d_bbox_dict(self, bbox3d_dict, T_world_rgbCam):
        """
        This function processes 3D bounding box data from a given dictionary,
        extracting dimensions, calculating depths, and computing transformation
        matrices relative to the camera.
        """
        bb3d_dimensions = bbox3d_dict["object_dimensions"]

        bb3d_depths_list = []
        Ts_world_object_list = []
        Ts_cam_object_list = []
        for _, pose_as_tensor in enumerate(bbox3d_dict["ts_world_object"]):
            T_world_object = SE3.from_matrix3x4(pose_as_tensor.numpy())
            T_cam_object = T_world_rgbCam.inverse() @ T_world_object

            # Add to lists
            Ts_world_object_list.append(
                torch.tensor(T_world_object.to_matrix3x4(), dtype=torch.float32)
            )

            Ts_cam_object_list.append(
                torch.tensor(T_cam_object.to_matrix3x4(), dtype=torch.float32)
            )
            bb3d_depths_list.append(T_cam_object.translation()[:, 2].item())

        # Convert lists to tensors
        bb3d_depths = torch.tensor(bb3d_depths_list, dtype=torch.float32)
        Ts_world_object = torch.stack(Ts_world_object_list, dim=0)
        Ts_cam_object = torch.stack(Ts_cam_object_list, dim=0)

        return bb3d_dimensions, bb3d_depths, Ts_world_object, Ts_cam_object

    def _update_gt_data_in_sample(self, atek_wds_sample, sample):
        """
        updates the sample dictionary with filtered ground truth data for both 2D and 3D bounding boxes.
        """
        from detectron2.structures import Boxes, Instances

        bbox2d_dict = atek_wds_sample["gt_data"]["obb2_gt"]["camera-rgb"]
        bbox3d_dict = atek_wds_sample["gt_data"]["obb3_gt"]["camera-rgb"]

        # Instance id between obb3 and obb2 should be the same
        assert torch.allclose(
            bbox3d_dict["instance_ids"], bbox2d_dict["instance_ids"], atol=0
        ), "instance ids in obb2 and obb3 needs to be exactly the same!"

        category_ids = bbox3d_dict["category_ids"]

        T_world_rgbCam = SE3.from_matrix3x4(sample["T_world_camera"])

        bb2ds_x0y0x1y1, bb2ds_area = self._process_2d_bbox_dict(bbox2d_dict)
        bb3d_dimensions, bb3d_depths, Ts_world_object, Ts_cam_object = (
            self._process_3d_bbox_dict(bbox3d_dict, T_world_rgbCam)
        )

        # Filter 1: ignore category = -1, meaning "Other".
        category_id_filter = category_ids > 0  # filter out -1 category = "Other"

        # Filter 2: ignore bboxes with small area
        bb2d_area_filter = bb2ds_area > self.min_bb2d_area

        # Filter 3: ignore bboxes with small depth
        bb3d_depth_filter = (self.min_bb3d_depth <= bb3d_depths) & (
            bb3d_depths <= self.max_bb3d_depth
        )

        # Combine all filters
        final_filter = category_id_filter & bb2d_area_filter & bb3d_depth_filter

        # Apply filter to create instances
        image_height = sample["height"]
        image_width = sample["width"]
        instances = Instances((image_height, image_width))
        instances.gt_classes = category_ids[final_filter]
        instances.gt_boxes = Boxes(bb2ds_x0y0x1y1[final_filter])

        # Create 3D bboxes
        Ts_cam_object_filtered = Ts_cam_object[final_filter]
        trans_cam_object_filtered = Ts_cam_object_filtered[:, :, 3]
        k_matrix = sample["K_matrix"]
        filtered_projection_2d = (
            k_matrix.repeat(len(trans_cam_object_filtered), 1, 1)
            @ trans_cam_object_filtered.unsqueeze(-1)
        ).squeeze(-1)
        filtered_projection_2d = filtered_projection_2d[:, :2] / filtered_projection_2d[
            :, 2
        ].unsqueeze(-1)
        instances.gt_boxes3D = torch.cat(
            [
                filtered_projection_2d,  # [N, 2]
                bb3d_depths[final_filter].unsqueeze(-1).clone().detach(),  # [N, 1]
                # Omni3d has the inverted zyx dimensions
                # https://github.com/facebookresearch/omni3d/blob/main/cubercnn/util/math_util.py#L144C1-L181C40
                bb3d_dimensions[final_filter].flip(-1).clone().detach(),  # [N, 3]
                trans_cam_object_filtered,  # [N, 3]
            ],
            axis=-1,
        )
        instances.gt_poses = Ts_cam_object_filtered[:, :, :3].clone().detach()

        # Update sample with filtered instance data
        sample["instances"] = instances
        sample["Ts_world_object"] = Ts_world_object[final_filter].clone().detach()
        sample["object_dimensions"] = bb3d_dimensions[final_filter].clone().detach()
        sample["category"] = category_ids[final_filter].clone().detach()

    @staticmethod
    def cubercnn_gt_to_atek_gt(
        cubercnn_dict: Dict,
        T_world_camera_np: np.array,
        camera_label: str = "camera-rgb",
        cubercnn_id_to_atek_id: Optional[Dict[int, int]] = None,
    ) -> Optional[Dict]:
        """
        A helper data transform function to convert the model input (gt) dict, or output (prediction) dict from CubeRCNN format,
        back to ATEK GT dict format (defined in `obb_sample_builder`, which is effectively obb3_gt_processor + obb2_gt_processor)
        CubeRCNN model is ran only on one camera stream, so user should specific which camera stream to use. By default, it is "camera-rgb".
        """
        cubercnn_instances = cubercnn_dict["instances"]
        # Skip if no instances
        if len(cubercnn_instances) == 0:
            return None

        # Check if the cubercnn_dict is a prediction dict or gt dict. If it is a gt dict,
        # "transfer" it to a prediction dict by filling in the pred fields. TODO: Consider another way to handle this!
        pred_flag = hasattr(cubercnn_instances, "pred_classes")
        if not pred_flag:
            # fake pred fields using gt fields
            num_instances = len(cubercnn_instances.gt_classes)
            cubercnn_instances.pred_classes = cubercnn_instances.gt_classes
            cubercnn_instances.pred_boxes = cubercnn_instances.gt_boxes
            cubercnn_instances.pred_dimensions = cubercnn_instances.gt_boxes3D[:, 3:6]
            cubercnn_instances.pred_center_cam = cubercnn_instances.gt_boxes3D[:, 6:9]
            cubercnn_instances.pred_pose = cubercnn_instances.gt_poses
            cubercnn_instances.scores = torch.ones(num_instances, dtype=torch.float32)

        # initialize ATEK GT dict
        atek_dict = {
            "obb3_gt": {},
            "obb2_gt": {},
            "scores": cubercnn_instances.scores.detach().cpu(),  # tensor, shape: [num_instances], float32
        }
        atek_dict["obb3_gt"][camera_label] = {
            "instance_ids": None,
            "category_names": None,
            "category_ids": cubercnn_instances.pred_classes.detach().cpu(),
        }
        atek_dict["obb2_gt"][camera_label] = {
            "instance_ids": None,
            "category_names": None,
            "category_ids": cubercnn_instances.pred_classes.detach().cpu(),
            "visibility_ratios": None,
        }

        # Fill in category ids
        if cubercnn_id_to_atek_id is not None:
            atek_id_list = [
                cubercnn_id_to_atek_id[id.item()]
                for id in cubercnn_instances.pred_classes
            ]
            atek_dict["obb3_gt"][camera_label]["category_ids"] = torch.tensor(
                atek_id_list, dtype=torch.int32
            )
            atek_dict["obb2_gt"][camera_label]["category_ids"] = torch.tensor(
                atek_id_list, dtype=torch.int32
            )
        else:
            atek_dict["obb3_gt"][camera_label]["category_ids"] = (
                cubercnn_instances.pred_classes.detach().cpu()
            )
            atek_dict["obb2_gt"][camera_label]["category_ids"] = (
                cubercnn_instances.pred_classes.detach().cpu()
            )

        # Fill category names
        atek_dict["obb3_gt"][camera_label]["category_names"] = [
            ATEK_CATEGORY_ID_TO_NAME[id.item()]
            for id in atek_dict["obb3_gt"][camera_label]["category_ids"]
        ]
        atek_dict["obb2_gt"][camera_label]["category_names"] = [
            ATEK_CATEGORY_ID_TO_NAME[id.item()]
            for id in atek_dict["obb2_gt"][camera_label]["category_ids"]
        ]

        # CubeRCNN dimensions are in reversed order (zyx) compared to ATEK (xyz)
        bbox3d_dim = (
            cubercnn_instances.pred_dimensions.detach().cpu()
        )  # tensor, shape [num_instances, 3]
        atek_dict["obb3_gt"][camera_label]["object_dimensions"] = torch.flip(
            bbox3d_dim, dims=[1]
        )

        # Fill in pose
        rotations = cubercnn_instances.pred_pose.detach().cpu()  # [num_instances, 3, 3]
        translations = (
            cubercnn_instances.pred_center_cam.detach().cpu().unsqueeze(2)
        )  # [num_instances, 3, 1]

        Ts_cam_object = SE3.from_matrix3x4(
            torch.cat((rotations, translations), dim=2).numpy()
        )
        T_world_cam = SE3.from_matrix3x4(T_world_camera_np)

        Ts_world_object = T_world_cam @ Ts_cam_object
        Ts_world_object = SE3.to_matrix3x4(Ts_world_object)  # [num_instances, 3, 4]
        if Ts_world_object.shape == (3, 4):
            Ts_world_object = Ts_world_object.reshape(1, 3, 4)
        atek_dict["obb3_gt"][camera_label]["ts_world_object"] = torch.tensor(
            Ts_world_object, dtype=torch.float32
        )

        # Fill in 2d bbox ranges
        bbox2d = (
            cubercnn_instances.pred_boxes.tensor.detach().cpu()
        )  # tensor, shape [num_instances, 4]
        # x0,y0,x1,y1 -> x0,x1,y0,y1
        atek_dict["obb2_gt"][camera_label]["box_ranges"] = torch.stack(
            (bbox2d[:, 0], bbox2d[:, 2], bbox2d[:, 1], bbox2d[:, 3]), dim=1
        )

        return atek_dict


def cubercnn_collation_fn(batch):
    # Simply collate as a list
    batch = list(batch)

    batch_o = defaultdict(list)

    for d in batch:
        for key, value in d.items():
            if torch.is_tensor(value):
                if batch_o[key] != []:
                    batch_o[key] = torch.cat([batch_o[key], value.unsqueeze(0)], dim=0)
                else:
                    batch_o[key] = value.unsqueeze(0)
            else:
                batch_o[key].append(value)
    return batch_o


def load_atek_wds_dataset_as_cubercnn(
    urls: List, batch_size: Optional[int], repeat_flag: bool, shuffle_flag: bool = False
) -> wds.WebDataset:
    cubercnn_model_adaptor = CubeRCNNModelAdaptor()

    return load_atek_wds_dataset(
        urls,
        batch_size=batch_size,
        dict_key_mapping=CubeRCNNModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(cubercnn_model_adaptor.atek_to_cubercnn)(),
        collation_fn=cubercnn_collation_fn,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )


def create_atek_dataloader_as_cubercnn(
    urls: List[str],
    batch_size: Optional[int] = None,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    wds_dataset = load_atek_wds_dataset_as_cubercnn(
        urls,
        batch_size=batch_size,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

    return torch.utils.data.DataLoader(
        wds_dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )


def get_tars(tar_yaml, relative_path: str = "", use_relative_path: bool = False):
    yaml_filename = os.path.basename(tar_yaml)
    if yaml_filename.startswith("streamable") or yaml_filename.startswith("local_"):
        # logger.info(f"Loading new-format yaml file.")
        tar_files = load_yaml_and_extract_tar_list(yaml_path=tar_yaml)
        return tar_files

    with open(tar_yaml, "r") as f:
        tar_files = yaml.safe_load(f)["tars"]
    if use_relative_path:
        if relative_path == "":
            data_dir = os.path.dirname(tar_yaml)
        else:
            data_dir = relative_path
        tar_files = [os.path.join(data_dir, x) for x in tar_files]
    return tar_files


def shuffle_tars(tar_list: List, shuffle_seed: int = 42):
    random.seed(shuffle_seed)
    random.shuffle(tar_list)
    return tar_list


def build_test_loader(cfg):
    # rank = comm.get_rank()
    # world_size = comm.get_world_size()

    # print("World size:", world_size)
    # print("Getting tars from rank:", rank)
    test_tars = get_tars(
        TEST_LIST, relative_path="", use_relative_path=False
    )

    test_tars_local = test_tars#[rank::world_size]
    local_batch_size = 2 #max(8 // world_size, 1)
    print("local_batch_size:", local_batch_size)

    test_wds = load_atek_wds_dataset_as_cubercnn(
        urls=test_tars_local,
        batch_size=local_batch_size,
        repeat_flag=False,
        shuffle_flag=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_wds,
        batch_size=None,
        num_workers=8,
        pin_memory=True,
    )

    dataset_name = os.path.basename(TEST_LIST).split(".")[0]
    MetadataCatalog.get(dataset_name).set(
        json_file="", image_root="", evaluator_type="coco"
    )

    return test_dataloader


def build_train_loader(shuffle_tars_flag: bool = False, shuffle_seed: int = 42):
    # rank = comm.get_rank()
    # world_size = comm.get_world_size()

    # print("World size:", world_size)
    # print("Getting tars from rank:", rank)
    train_tars = get_tars(
        TRAIN_LIST, relative_path="", use_relative_path=False
    )

    train_tars_local = train_tars#[rank::world_size]
    local_batch_size = 4
    print("local_batch_size:", local_batch_size)

    # Perform random shuffle to local tars
    if shuffle_tars_flag:
        train_tars_local = shuffle_tars(
            tar_list=train_tars_local, shuffle_seed=shuffle_seed
        )

    train_wds = load_atek_wds_dataset_as_cubercnn(
        urls=train_tars_local,
        batch_size=local_batch_size,
        repeat_flag=False,
        shuffle_flag=True,  # always perform local shuffle
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_wds,
        batch_size=None,
        num_workers=8,
        pin_memory=True,
    )

    dataset_name = os.path.basename(TRAIN_LIST).split(".")[0]
    MetadataCatalog.get(dataset_name).set(
        json_file="", image_root="", evaluator_type="coco"
    )

    return train_dataloader