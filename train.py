"""
Note that batch_size refers to the batch_size per GPU.

accelerate launch train.py training.batch_size=8 training.max_iterations=450000
"""

import datetime
import os
import random
import socket
import time
from glob import glob

import hydra
import ipdb  # noqa: F401
import numpy as np
import omegaconf
import torch
from accelerate import Accelerator
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.fisheyecameras import FishEyeCameras

import wandb
from ray_diffusion.dataset.co3d_v2 import Co3dDataset
from ray_diffusion.dataset.tartanair import TartanAir, AirSampler
from ray_diffusion.dataset.megadepth import MegaDepth
from ray_diffusion.dataset.atek_adaptor import build_train_loader

from ray_diffusion.model.diffuser import RayDiffuser
from ray_diffusion.model.scheduler import NoiseScheduler
from ray_diffusion.utils.normalize import normalize_cameras_batch
from ray_diffusion.utils.rays import cameras_to_rays
from ray_diffusion.utils.visualization import (
    create_plotly_cameras_visualization,
    create_training_visualizations,
)
from ray_diffusion.eval.utils import angle_btw

os.umask(000)  # Default to 777 permissions


class Trainer(object):
    def __init__(self, cfg):
        print("config", cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.debug = cfg.debug
        self.resume = cfg.training.resume
        self.pretrain_path = cfg.training.pretrain_path

        self.batch_size = cfg.training.batch_size
        self.max_iterations = cfg.training.max_iterations
        self.mixed_precision = cfg.training.mixed_precision
        self.interval_visualize = cfg.training.interval_visualize
        self.interval_save_checkpoint = cfg.training.interval_save_checkpoint
        self.interval_delete_checkpoint = cfg.training.interval_delete_checkpoint
        self.interval_evaluate = cfg.training.interval_evaluate
        self.delete_all = cfg.training.delete_all_checkpoints_after_training
        self.freeze_encoder = cfg.training.freeze_encoder
        self.translation_scale = cfg.training.translation_scale
        self.num_visualize = 2
        self.regression = cfg.training.regression
        self.load_extra_cameras = cfg.training.load_extra_cameras
        self.calculate_intrinsics = cfg.training.calculate_intrinsics
        self.normalize_first_camera = cfg.training.normalize_first_camera

        self.model_type = cfg.model.model_type
        self.pred_x0 = cfg.model.pred_x0
        self.num_patches_x = cfg.model.num_patches_x
        self.num_patches_y = cfg.model.num_patches_y
        self.depth = cfg.model.depth
        self.num_images = cfg.model.num_images
        self.random_num_images = cfg.model.random_num_images
        self.feature_extractor = cfg.model.feature_extractor
        self.append_ndc = cfg.model.append_ndc

        self.dataset_name = cfg.dataset.name
        self.category = cfg.dataset.category
        self.apply_augmentation = cfg.dataset.apply_augmentation

        if self.regression:
            assert self.pred_x0

        self.start_time = None
        self.iteration = 0
        self.epoch = 0
        self.wandb_id = None
        self.hostname = socket.gethostname()

        self.accelerator = Accelerator(
            even_batches=False,
            device_placement=False,
        )
        self.device = self.accelerator.device

        scheduler = NoiseScheduler(
            type=cfg.noise_scheduler.type,
            max_timesteps=cfg.noise_scheduler.max_timesteps,
            beta_start=cfg.noise_scheduler.beta_start,
            beta_end=cfg.noise_scheduler.beta_end,
        )

        self.model = RayDiffuser(
            depth=self.depth,
            width=self.num_patches_x,
            P=1,
            max_num_images=self.num_images,
            noise_scheduler=scheduler,
            freeze_encoder=self.freeze_encoder,
            feature_extractor=self.feature_extractor,
            append_ndc=self.append_ndc,
        ).to(self.device)
        self.lr = 1e-4
        self.optimizer = torch.optim.Adam(self.model.ray_predictor.parameters(), lr=self.lr)
        self.optimizer_flow = torch.optim.Adam(self.model.distortion_predictor.parameters(), lr=self.lr)
        self.gradscaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        if self.dataset_name == "co3d":
            self.dataset = Co3dDataset(
                category=self.category,
                split="train",
                num_images=self.num_images,
                apply_augmentation=self.apply_augmentation,
                load_extra_cameras=self.load_extra_cameras,
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        elif self.dataset_name == "tartanair":
            self.dataset = TartanAir(scale=1, augment=True)
            self.train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        elif self.dataset_name == "megadepth":
            self.dataset = MegaDepth(
                split="train",
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        elif self.dataset_name == "aria":
            self.train_dataloader = build_train_loader(
                shuffle_tars_flag=True,
            )

        self.model, self.optimizer, self.train_dataloader, self.optimizer_flow = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.optimizer_flow
        )

        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.accelerator.is_main_process:
            name = os.path.basename(self.output_dir)
            name += f"_{self.category}"
            name += f"_{self.model_type}"
            name += f"_B{self.batch_size * self.accelerator.num_processes}"
            name += f"_N{self.num_images}"
            if self.random_num_images:
                name += "Rand"
            name += f"_D{self.depth}"
            name += f"_LR{self.lr}"
            name += f"_T{scheduler.max_timesteps}"
            if self.num_patches_x != 16 or self.num_patches_y != 16:
                name += f"_P{self.num_patches_x}x{self.num_patches_y}"
            if self.mixed_precision:
                name += "_AMP"
            if self.pred_x0:
                name += "_predX0"
                if self.regression:
                    name += "reg"
            if not self.freeze_encoder:
                name += "_FTEnc"
            if self.pretrain_path != "":
                name += "_Pretrained"
            else:
                if self.feature_extractor != "dino":
                    name += f"_{self.feature_extractor}"
            if self.normalize_first_camera:
                name += "_NormFirst"

            print("Output dir:", self.output_dir)
            with open(os.path.join(self.output_dir, name), "w"):
                # Create empty tag with name
                pass
            self.name = name

            conf_dict = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            conf_dict["output_dir"] = self.output_dir
            conf_dict["hostname"] = self.hostname

        print("Resuming from checkpoint. ")
        # self.load_model("/ocean/projects/cis240055p/liuyuex/gemma/DistortCalib3/pretrain/models/co3d_diffusion/checkpoints/ckpt_00450000.pth", load_metadata=False)
        # self.load_model("/ocean/projects/cis240055p/liuyuex/gemma/DistortCalib/output_3x3_world_no_edge2/checkpoints/ckpt_00035000.pth", load_metadata=False)
        self.load_model("/ocean/projects/cis240055p/liuyuex/gemma/DistortCalib3/output_co3d_distort/checkpoints/ckpt_00050000.pth", load_metadata=False)
        self.iteration = 0

        if self.resume:
            checkpoint_files = sorted(glob(os.path.join(self.checkpoint_dir, "*.pth")))
            last_checkpoint = checkpoint_files[-1]
            print("Resuming from checkpoint:", last_checkpoint)
            self.load_model(last_checkpoint, load_metadata=True)
        elif self.pretrain_path != "":
            print("Loading pretrained model:", self.pretrain_path)
            self.load_model(self.pretrain_path, load_metadata=False)

        if self.accelerator.is_main_process:
            mode = "online" if cfg.debug.wandb else "disabled"
            if self.wandb_id is None:
                self.wandb_id = wandb.util.generate_id()
            self.wandb_run = wandb.init(
                mode=mode,
                name=name,
                project=cfg.debug.project_name,
                config=conf_dict,
                resume=self.resume,
                id=self.wandb_id,
            )
            wandb.define_metric("iteration")
            noise_schedule = self.get_module().noise_scheduler.plot_schedule(
                return_image=True
            )
            wandb.log(
                {"Schedule": wandb.Image(noise_schedule, caption="Noise Schedule")}
            )

    def get_module(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    def train(self):
        while self.iteration < self.max_iterations:
            for batch in self.train_dataloader:
                t0 = time.time()
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    images = batch["image_d"].to(self.device)
                    # if self.dataset_name == "co3d":
                    focal_lengths = batch["focal_length"].to(self.device)
                    crop_params = batch["crop_parameters"].to(self.device)
                    principal_points = batch["principal_point"].to(self.device)
                    R = batch["R"].to(self.device)
                    T = batch["T"].to(self.device)
                    flows = batch["flow"].to(self.device)

                    if self.dataset_name == "aria":
                        radial_params = batch["radial_params"].to(self.device)
                        tangential_params = batch["tangential_params"].to(self.device)
                        thin_prism_params = batch["thinprism_params"].to(self.device)
                        cameras_og = [
                                FishEyeCameras(
                                    focal_length=focal_lengths[b],
                                    principal_point=principal_points[b],
                                    R=R[b],
                                    T=T[b],
                                    radial_params=radial_params[b],
                                    tangential_params=tangential_params[b],
                                    thin_prism_params=thin_prism_params[b],
                                    device=self.device,
                                )
                                for b in range(self.batch_size)
                            ]
                    else:
                        cameras_og = [
                            PerspectiveCameras(
                                focal_length=focal_lengths[b],
                                principal_point=principal_points[b],
                                R=R[b],
                                T=T[b],
                                device=self.device,
                            )
                            for b in range(self.batch_size)
                        ]

                    if self.num_images == 1 or self.dataset_name == "aria":
                        cameras = cameras_og
                    else:
                        cameras, _ = normalize_cameras_batch(
                            cameras=cameras_og,
                            scale=self.translation_scale,
                            normalize_first_camera=self.normalize_first_camera,
                        )

                    # Now that cameras are normalized, fix shapes of camera parameters
                    if self.load_extra_cameras or self.random_num_images:
                        if self.random_num_images:
                            num_images = torch.randint(2, self.num_images + 1, (1,))
                        else:
                            num_images = self.num_images

                        # The correct number of images is already loaded.
                        # Only need to modify these camera parameters shapes.
                        crop_params = crop_params[:, :num_images]
                        R = R[:, :num_images]
                        T = T[:, :num_images]
                        images = images[:, :num_images]
                        flows = flows[:, :num_images]

                        if self.dataset_name == "aria":
                            cameras = [
                                FishEyeCameras(
                                    focal_length=cameras[b].focal[:num_images],
                                    principal_point=cameras[b].principal_point[:num_images],
                                    R=cameras[b].R[:num_images],
                                    T=cameras[b].T[:num_images],
                                    radial_params=cameras[b].radial_params[:num_images],
                                    tangential_params=cameras[b].tangential_params[:num_images],
                                    thin_prism_params=cameras[b].thin_prism_params[:num_images],
                                    # device=self.device,
                                )
                                for b in range(self.batch_size)
                            ]
                        else:
                            cameras = [
                                PerspectiveCameras(
                                    focal_length=cameras[b].focal_length[:num_images],
                                    principal_point=cameras[b].principal_point[:num_images],
                                    R=cameras[b].R[:num_images],
                                    T=cameras[b].T[:num_images],
                                    device=self.device,
                                )
                                for b in range(self.batch_size)
                            ]

                    if self.regression:
                        low = self.get_module().noise_scheduler.max_timesteps - 1
                    else:
                        low = 0

                    t = torch.randint(
                        low=low,
                        high=self.get_module().noise_scheduler.max_timesteps,
                        size=(self.batch_size,),
                        device=self.device,
                    )

                    rays = []
                    for camera, crop_param, flow in zip(cameras, crop_params, flows):
                        # try:
                        #     print("flow", flow.shape)
                        # except:
                        #     import pdb; pdb.set_trace()
                        if self.dataset_name == "aria":
                            r = cameras_to_rays(
                                cameras=camera,
                                num_patches_x=self.num_patches_x,
                                num_patches_y=self.num_patches_y,
                                crop_parameters=crop_param,
                            )
                        else:
                            r = cameras_to_rays(
                                cameras=camera,
                                num_patches_x=self.num_patches_x,
                                num_patches_y=self.num_patches_y,
                                crop_parameters=crop_param,
                                flow=flow,
                            )
                        rays.append(
                            r.to_spatial(include_ndc_coordinates=self.append_ndc)
                        )
                    rays_tensor = torch.stack(rays, dim=0)
                    if self.append_ndc:
                        ndc_coordinates = rays_tensor[..., -2:, :, :]
                        rays_tensor = rays_tensor[..., :-2, :, :]  # (B, N, 6, H, W)
                    else:
                        ndc_coordinates = None
                    eps_pred, eps = self.model(
                        images=images,
                        rays=rays_tensor,
                        t=t,
                        ndc_coordinates=ndc_coordinates,
                    )
                    if self.pred_x0:
                        target = rays_tensor
                    else:
                        target = eps
                    loss = torch.mean((eps_pred - target) ** 2)

                    # Angular loss for camera parameters
                    v1 = eps_pred.reshape(-1, 6)
                    v2 = target.reshape(-1, 6)
                    loss_angular = [angle_btw(v1[i], v2[i]) for i in range(len(v1))]
                    loss_angular = torch.nan_to_num(torch.stack(loss_angular), 0.0).mean()
                    loss = loss + loss_angular * 0.8

                    # flow loss
                    flow_pred = self.model(rays=eps_pred, distort_pred=True)
                    loss_flow = torch.mean((flow_pred - flows) ** 2)
                    loss += loss_flow

                if self.mixed_precision:
                    self.gradscaler.scale(loss).backward()
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.step(self.optimizer_flow)
                    self.gradscaler.update()
                else:
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer_flow.step()

                if self.accelerator.is_main_process:
                    if self.iteration % 10 == 0:
                        self.log_info(loss, loss_angular, loss_flow, t0)

                    if self.iteration % self.interval_visualize == 0:
                        self.visualize(
                            images=images,
                            cameras_gt=cameras,
                            crop_parameters=crop_params,
                            flows=flow_pred.detach().clone(),
                        )

                if self.accelerator.is_main_process:
                    if self.iteration % self.interval_save_checkpoint == 0:
                        self.save_model()

                    if self.iteration % self.interval_delete_checkpoint == 0:
                        self.clear_old_checkpoints(self.checkpoint_dir)

                    if self.iteration >= self.max_iterations + 1:
                        if self.delete_all:
                            self.clear_old_checkpoints(
                                self.checkpoint_dir, clear_all_old=True
                            )
                        return
                self.iteration += 1
            self.epoch += 1

    def load_model(self, path, load_metadata=True):
        current_model_dict = self.get_module().state_dict()
        save_dict = torch.load(path, map_location=self.device)["state_dict"]
        new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), save_dict.values())}

        missing, unexpected = self.get_module().load_state_dict(
            new_state_dict,
            strict=False,
        )
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        if load_metadata:
            self.iteration = save_dict["iteration"]
            self.epoch = save_dict["epoch"]
            time_elapsed = save_dict["elapsed"]
            self.start_time = time.time() - time_elapsed
            if "wandb_id" in save_dict:
                self.wandb_id = save_dict["wandb_id"]
            self.optimizer.load_state_dict(save_dict["optimizer"])
            self.optimizer_flow.load_state_dict(save_dict["optimizer_flow"])
            self.gradscaler.load_state_dict(save_dict["gradscaler"])

    def save_model(self):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{self.iteration:08d}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        save_dict = {
            "epoch": self.epoch,
            "elapsed": elapsed,
            "gradscaler": self.gradscaler.state_dict(),
            "iteration": self.iteration,
            "state_dict": self.get_module().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_flow": self.optimizer_flow.state_dict(),
            "wandb_id": self.wandb_id,
        }
        torch.save(save_dict, path)

    def clear_old_checkpoints(self, checkpoint_dir, clear_all_old=False):
        print("Clearing old checkpoints")
        checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "ckpt_*.pth")))
        if clear_all_old:
            for checkpoint_file in checkpoint_files[:-1]:
                os.remove(checkpoint_file)
        else:
            for checkpoint_file in checkpoint_files:
                checkpoint = os.path.basename(checkpoint_file)
                checkpoint_iteration = int("".join(filter(str.isdigit, checkpoint)))
                if checkpoint_iteration % self.interval_delete_checkpoint != 0:
                    os.remove(checkpoint_file)

    def log_info(self, loss, loss_angular, loss_flow, t0):
        if self.start_time is None:
            self.start_time = time.time()
        time_elapsed = round(time.time() - self.start_time)
        time_remaining = round(
            (time.time() - self.start_time)
            / (self.iteration + 1)
            * (self.max_iterations - self.iteration)
        )
        disp = [
            f"Iter: {self.iteration}/{self.max_iterations}",
            f"Epoch: {self.epoch}",
            f"Loss: {loss.item():.4f}",
            f"Loss_angular: {loss_angular.item():.4f}",
            f"Loss_flow: {loss_flow.item():.4f}",
            f"Elap: {str(datetime.timedelta(seconds=time_elapsed))}",
            f"Rem: {str(datetime.timedelta(seconds=time_remaining))}",
            self.hostname,
            self.name,
        ]
        print(", ".join(disp), flush=True)
        wandb.log(
            {
                "loss": loss.item(),
                "loss_angular": loss_angular.item(),
                "loss_flow": loss_flow.item(),
                "iter_time": time.time() - t0,
                "lr": self.lr,
                "iteration": self.iteration,
                "hours_remaining": time_remaining / 3600,
            }
        )

    def visualize(self, images, cameras_gt, flows, crop_parameters=None):
        self.get_module().eval()
        for camera in cameras_gt:
            # AMP may not cast back to float
            camera.R = camera.R.float()
            camera.T = camera.T.float()

        vis_images, cameras_pred_batched = create_training_visualizations(
            model=self.get_module(),
            images=images[: self.num_visualize],
            device=self.device,
            cameras_gt=cameras_gt,
            pred_x0=self.pred_x0,
            num_images=images.shape[1],
            crop_parameters=crop_parameters[: self.num_visualize],
            # visualize_pred=self.regression,
            # return_first=self.regression,
            visualize_pred=True,
            return_first=True,
            calculate_intrinsics=self.calculate_intrinsics,
            flows=flows[: self.num_visualize],
        )

        for i, cameras_pred in enumerate(cameras_pred_batched):
            fig = create_plotly_cameras_visualization(cameras_gt[i], cameras_pred, i)
            plot = wandb.Plotly(fig)
            wandb.log({f"Vis {i} plotly": plot})

        for i, vis_image in enumerate(vis_images):
            im = wandb.Image(
                vis_image, caption=f"iteration {self.iteration} example {i}"
            )
            wandb.log({f"Vis {i}": im})
        self.get_module().train()


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(cfg):
    print(cfg)
    torch.autograd.set_detect_anomaly(cfg.debug.anomaly_detection)
    torch.set_float32_matmul_precision(cfg.training.matmul_precision)
    trainer = Trainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
