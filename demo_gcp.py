import argparse
import base64
import io
import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from ray_diffusion.dataset import CustomDataset
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras
from ray_diffusion.utils.visualization import view_color_coded_images_from_tensor
from ray_diffusion.utils.visualization import plot_arrow_on_image, unnormalize_image

HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
<body><img src="data:image/png;charset=utf-8;base64,{image_encoded}"/>
{plotly_html}</body></html>"""


def get_parser():
    parser = argparse.ArgumentParser()
    # /gcs/xlouise-xcloud-us-central1/distortcalib/outputs_wo_norm
    parser.add_argument("--image_dir", type=str, default="/workdir/DistortCalib3/examples/kitchen/images")
    parser.add_argument("--model_dir", type=str, default="/gcs/xlouise-xcloud-us-central1/")
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--bbox_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="/gcs/xlouise-xcloud-us-central1/output_vis_kitchen_distort/output_cameras.html")
    return parser



def plotly_scene_visualization(R_pred, T_pred):
    num_frames = len(R_pred)

    camera = {}
    for i in range(num_frames):
        camera[i] = PerspectiveCameras(R=R_pred[i, None], T=T_pred[i, None])

    fig = plot_scene(
        {"scene": camera},
        camera_scale=0.03,
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")
    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
    return fig


def main(image_dir, model_dir, mask_dir, bbox_path, output_path):
    device = torch.device("cuda:0")
    model, cfg = load_model(model_dir, device=device)
    if osp.exists(bbox_path):
        bboxes = json.load(open(bbox_path))
    else:
        bboxes = None
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        bboxes=bboxes,
        mask_images=False,
    )
    num_frames = dataset.n
    batch = dataset.get_data(ids=np.arange(num_frames))
    images = batch["image"].to(device)
    crop_params = batch["crop_params"].to(device)

    is_regression = cfg.training.regression
    if is_regression:
        # regression
        pred = predict_cameras(
            model=model,
            images=images,
            device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            use_regression=True,
        )
        predicted_cameras = pred[0]
    else:
        # diffusion
        patch_size = 32
        img_size = 448
        pred = predict_cameras(
            model=model,
            images=images,
            device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            additional_timesteps=(70,),  # We found that X0 at T=30 is best.
            rescale_noise="zero",
            use_regression=False,
            max_num_images=None if num_frames <= 8 else 8,  # Auto-batch for N > 8.
            pbar=True,
            return_rays=True,
            num_patches_x=patch_size,
            num_patches_y=patch_size,

        )
        # predicted_cameras = pred[1][0]
        _, rays, predicted_cameras, _ = pred
        predicted_cameras = predicted_cameras[0]
        directions = rays.get_directions().permute(2, 0, 1)
        image = images[0]
        np.save("/gcs/xlouise-xcloud-us-central1/output_vis_kitchen_distort/directions.npy", directions.cpu().numpy())
        np.save("/gcs/xlouise-xcloud-us-central1/output_vis_kitchen_distort/image.npy", image.cpu().numpy())
        for i in range(num_frames):
            img = images[i].cpu().numpy()
            img = unnormalize_image(img)
            plt.clf()
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(f"/gcs/xlouise-xcloud-us-central1/output_vis_kitchen_distort/image_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            arrow_field = directions[:, i, :].reshape((3, patch_size, patch_size))
            ray_torch = torch.tensor(arrow_field)
            ray_torch = torch.nn.functional.interpolate(ray_torch.unsqueeze(0), size=[img_size, img_size])
            ray_torch_gt = torch.rand(ray_torch.shape) * 0.005 + ray_torch
            plot_arrow_on_image(img, ray_torch.squeeze().cpu().numpy(), sparse_factor=20, save_path=f"/gcs/xlouise-xcloud-us-central1/output_vis_kitchen_distort/directions_{i}.png")
            plt.clf()
            plt.close()
            plot_arrow_on_image(img, ray_torch_gt.squeeze().cpu().numpy(), sparse_factor=20, save_path=f"/gcs/xlouise-xcloud-us-central1/output_vis_kitchen_distort/directions_gt_{i}.png")


    # Visualize cropped and resized images
    fig = plotly_scene_visualization(predicted_cameras.R, predicted_cameras.T)
    html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    s = io.BytesIO()
    view_color_coded_images_from_tensor(images)
    plt.savefig(s, format="png", bbox_inches="tight")
    plt.close()
    image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    with open(output_path, "w") as f:
        s = HTML_TEMPLATE.format(
            image_encoded=image_encoded,
            plotly_html=html_plot,
        )
        f.write(s)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
