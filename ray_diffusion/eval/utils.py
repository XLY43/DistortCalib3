import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras
from ray_diffusion.utils.rays import cameras_to_rays

def angle_btw(v1, v2):
    u1 = v1 / torch.linalg.norm(v1)
    u2 = v2 / torch.linalg.norm(v2)
    y = u1 - u2
    x = u1 + u2
    a0 = 2 * torch.arctan(torch.linalg.norm(y) / torch.linalg.norm(x))
    if (not torch.signbit(a0)) or torch.signbit(torch.tensor(math.pi).to(v1.device) - a0):
        return a0
    elif torch.signbit(a0):
        return 0.0
    else:
        return torch.tensor(math.pi).to(v1.device)


def full_scene_scale(batch):
    """
    Recovers the scale of the scene, defined as the distance between the centroid of
    the cameras to the furthest camera.

    Args:
        batch (dict): batch containing the camera parameters for all cameras in the
            sequence.

    Returns:
        float: scale of the scene.
    """
    cameras = PerspectiveCameras(R=batch["R"], T=batch["T"])
    cc = cameras.get_camera_center()
    centroid = torch.mean(cc, dim=0)

    diffs = cc - centroid
    norms = torch.linalg.norm(diffs, dim=1)

    furthest_index = torch.argmax(norms).item()
    scale = norms[furthest_index].item()
    return scale


def get_permutations(num_images):
    permutations = []
    for i in range(0, num_images):
        for j in range(0, num_images):
            if i != j:
                permutations.append((j, i))

    return permutations


def n_to_np_rotations(num_frames, n_rots):
    R_pred_rel = []
    permutations = get_permutations(num_frames)
    for i, j in permutations:
        R_pred_rel.append(n_rots[i].T @ n_rots[j])
    R_pred_rel = torch.stack(R_pred_rel)

    return R_pred_rel


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation2, rotation1.transpose(0, 2, 1))
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


# A should be GT, B should be predicted
def compute_optimal_alignment(A, B):
    """
    Compute the optimal scale s, rotation R, and translation t that minimizes:
    || A - (s * B @ R + T) || ^ 2

    Reference: Umeyama (TPAMI 91)

    Args:
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        s (float): scale.
        R (torch.Tensor): rotation matrix (3, 3).
        t (torch.Tensor): translation (3,).
    """
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    # normally with R @ B, this would be A @ B.T
    H = (B - B_bar).T @ (A - A_bar)
    U, S, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    variance = torch.sum((B - B_bar) ** 2)
    scale = 1 / variance * torch.trace(torch.diag(S) @ S_prime)
    R = U @ S_prime @ Vh
    t = A_bar - scale * B_bar @ R

    A_hat = scale * B @ R + t
    return A_hat, scale, R, t


def compute_camera_center_error(R_pred, T_pred, R_gt, T_gt, gt_scene_scale):
    cameras_gt = PerspectiveCameras(R=R_gt, T=T_gt)
    cc_gt = cameras_gt.get_camera_center()
    cameras_pred = PerspectiveCameras(R=R_pred, T=T_pred)
    cc_pred = cameras_pred.get_camera_center()

    A_hat, _, _, _ = compute_optimal_alignment(cc_gt, cc_pred)
    norm = torch.linalg.norm(cc_gt - A_hat, dim=1) / gt_scene_scale

    norms = np.ndarray.tolist(norm.detach().cpu().numpy())
    return norms, A_hat

def compute_ray_direction_error(rays_pred, R_gt, T_gt, flow, patch_size=16):
    cameras_gt = PerspectiveCameras(R=R_gt, T=T_gt).to('cpu')
    gt_rays = cameras_to_rays(cameras=cameras_gt.to('cpu'),num_patches_x=patch_size,num_patches_y=patch_size,crop_parameters=None,flow=flow.to('cpu'))
    # Rays_error = torch.mean((gt_rays.get_directions().to(rays_pred.device) - rays_pred.get_directions()) ** 2)
    
    # Angular loss for camera parameters
    v1 = rays_pred.get_directions()
    v2 = gt_rays.get_directions().to(rays_pred.device)
    loss_angular = [angle_btw(v1[i], v2[i]) for i in range(len(v1))]
    Rays_error = torch.nan_to_num(torch.stack(loss_angular), 0.0).mean()
    return Rays_error
