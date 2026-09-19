# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for CasaMaestro

CasaMaestro (https://github.com/george-attano/CasaMaestro) is a house-scale
multi-view panorama reconstruction model built on top of Depth Anything 3 (DA3).
It reuses the DA3 network definition verbatim and only swaps the camera decoder
for a panorama-specific one, so this wrapper reuses the ``depth_anything_3``
package installed via the ``depth-anything-3`` optional dependency and only
vendors the extra panoramic camera decoder (see ``cam_pano.py``).

Key difference w.r.t. the DA3 wrapper: CasaMaestro consumes equirectangular
panoramas and predicts *metric distance along the viewing ray* for every pixel
of the equirectangular grid. There is no pinhole intrinsics matrix for such an
input, so the unified outputs are built from analytic equirectangular ray
directions instead of ``intrinsics``.
"""

import math

import torch
from depth_anything_3.cfg import create_object, load_config
from omegaconf import OmegaConf

from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.device import (
    get_amp_dtype,
    get_autocast_device_type,
    get_device,
)
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
)

# Camera decoder used by the panorama checkpoints of CasaMaestro
PANO_CAM_DEC = {
    "__object__": {
        "path": "mapanything.models.external.casamaestro.cam_pano",
        "name": "CameraDecPano",
        "args": "as_params",
    },
}


def equirectangular_ray_directions(height, width, device, dtype):
    """
    Compute unit ray directions for an equirectangular (panoramic) image grid.

    Uses the same spherical parameterization as the CasaMaestro reference
    implementation (``casameastro_vis.py``):
        phi   = (y / H - 0.5) * pi          (elevation, top row maps to -pi/2)
        theta = (x / W - 0.5) * 2 * pi      (azimuth)
        d = (cos(phi) * sin(theta), sin(phi), cos(phi) * cos(theta))

    Args:
        height (int): Height of the equirectangular grid.
        width (int): Width of the equirectangular grid.
        device (torch.device): Device to create the directions on.
        dtype (torch.dtype): Dtype to create the directions with.

    Returns:
        torch.Tensor: Unit ray directions of shape (H, W, 3).
    """
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    phi = (ys / height - 0.5) * math.pi
    theta = (xs / width - 0.5) * 2.0 * math.pi
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    cos_phi = torch.cos(phi)
    ray_directions = torch.stack(
        [
            cos_phi * torch.sin(theta),
            torch.sin(phi),
            cos_phi * torch.cos(theta),
        ],
        dim=-1,
    )

    # Already unit norm analytically; normalize to guard against float error
    return ray_directions / ray_directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def to_4x4(pose):
    """
    Convert (..., 3, 4) poses to homogeneous (..., 4, 4) poses.

    Args:
        pose (torch.Tensor): Pose tensor of shape (..., 3, 4) or (..., 4, 4).

    Returns:
        torch.Tensor: Pose tensor of shape (..., 4, 4).
    """
    if pose.shape[-2:] == (4, 4):
        return pose
    assert pose.shape[-2:] == (3, 4), (
        f"Unexpected pose shape {tuple(pose.shape)}, expected (...,3,4) or (...,4,4)"
    )
    last_row = torch.zeros_like(pose[..., :1, :])
    last_row[..., 0, 3] = 1.0

    return torch.cat([pose, last_row], dim=-2)


def cam2world_relative_to_first_view(cam2world):
    """
    Express cam2world poses relative to the first view, which becomes identity.

    CasaMaestro predicts poses up to a global rigid transform, and its reference
    inference scripts anchor the reconstruction to the first view. This also
    matches the MapAnything convention of using the first view as the reference
    frame.

    Args:
        cam2world (torch.Tensor): cam2world poses of shape (B, V, 4, 4).

    Returns:
        torch.Tensor: Re-anchored cam2world poses of shape (B, V, 4, 4).
    """
    world2cam_first = torch.linalg.inv(cam2world[:, 0])
    relative = world2cam_first.unsqueeze(1) @ cam2world
    relative[:, 0] = torch.eye(4, device=cam2world.device, dtype=cam2world.dtype)

    return relative


class CasaMaestroWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        da3_config="depth_anything_3.configs.da3-large",
        use_pano_cam_dec=True,
        cam_dec_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        self.da3_config = da3_config
        self.use_pano_cam_dec = use_pano_cam_dec

        # Build the DA3 backbone/head from the packaged DA3 config
        cfg = load_config(da3_config)

        # CasaMaestro's panorama checkpoints replace the DA3 camera decoder with
        # a cross-view attention decoder that regresses a unit quaternion and a
        # fixed (unused) FoV. The no-mask checkpoint keeps the original DA3 one.
        if self.use_pano_cam_dec:
            pano_cam_dec = dict(PANO_CAM_DEC)
            pano_cam_dec["dim_in"] = cfg.cam_dec.dim_in
            if cam_dec_kwargs is not None:
                pano_cam_dec.update(dict(cam_dec_kwargs))
            cfg.cam_dec = OmegaConf.create(pano_cam_dec)

        self.model = create_object(cfg)

        # Load the released CasaMaestro weights
        print(f"Loading CasaMaestro checkpoint from {self.ckpt_path} ...")
        state_dict = self._load_state_dict(self.ckpt_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(
            f"[CasaMaestro load] missing={len(missing)} unexpected={len(unexpected)}"
        )

        self.device = get_device()
        # Get the dtype for CasaMaestro inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = get_amp_dtype(self.device)

    @staticmethod
    def _load_state_dict(ckpt_path):
        """
        Load a CasaMaestro state dict from a ``.pt`` or ``.safetensors`` file.

        Args:
            ckpt_path (str): Path to the checkpoint.

        Returns:
            dict: The model state dict.
        """
        if str(ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = (
                ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            )

        # Strip the "model." prefix used by some released checkpoints
        if any(key.startswith("model.") for key in state_dict.keys()):
            state_dict = {
                key[len("model.") :]: value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }

        return state_dict

    def forward(self, views):
        """
        Forward pass wrapper for CasaMaestro

        Assumptions:
        - All the input views have the same image shape.
        - The input images are equirectangular panoramas.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["dinov2"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        _, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "dinov2", (
            "CasaMaestro expects DINOv2 normalization for the input images"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Run the CasaMaestro model
        # CasaMaestro regresses poses with its panoramic camera decoder, so the
        # ray-based pose estimation of DA3 is disabled.
        with torch.autocast(get_autocast_device_type(self.device), dtype=self.dtype):
            results = self.model(
                images,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                use_ray_pose=False,
            )

        # Need high precision for transformations
        with torch.autocast(get_autocast_device_type(self.device), enabled=False):
            # Convert the predicted world2cam poses to cam2world in the frame of
            # the first (reference) view
            pred_world2cam = to_4x4(results["extrinsics"].float())
            pred_cam2world = closed_form_inverse_se3(
                pred_world2cam.reshape(-1, 4, 4)
            ).reshape(pred_world2cam.shape)
            pred_cam2world = cam2world_relative_to_first_view(pred_cam2world)

            # Analytic equirectangular ray directions, shared across all views
            ray_directions = equirectangular_ray_directions(
                height, width, images.device, torch.float32
            )

            res = []
            for view_idx in range(num_views):
                # CasaMaestro predicts metric distance along the viewing ray
                curr_view_depth_along_ray = results["depth"][:, view_idx, ...].float()
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)
                curr_view_confidence = results["depth_conf"][:, view_idx, ...].float()

                # Expand the ray directions to the batch size of the view
                curr_view_ray_dirs = ray_directions.expand(
                    curr_view_depth_along_ray.shape[0], -1, -1, -1
                )

                # Get the camera frame pointmaps
                curr_view_pts3d_cam = curr_view_depth_along_ray * curr_view_ray_dirs

                # Convert the extrinsics to quaternions and translations
                curr_view_extrinsic = pred_cam2world[:, view_idx, ...]
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Get the pointmaps
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list
                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                    }
                )

        return res
