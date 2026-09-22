# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for PanoVGGT

PanoVGGT (https://github.com/YijingGuo-June/PanoVGGT) is a VGGT-style feed-forward
model for equirectangular panoramas: it predicts a per-pixel distance along the ERP
viewing ray plus a cam2world pose per view.

Like the CasaMaestro wrapper, and unlike the pinhole wrappers, the unified outputs are
built from analytic equirectangular ray directions rather than from an intrinsics matrix,
because an ERP image has no pinhole ``K``.

Note [panovggt-normalizes-internally]:
``Aggregator.forward()`` applies the ImageNet mean/std itself
(``images = (images - self._resnet_mean) / self._resnet_std``), so this wrapper expects
``data_norm_type == "identity"``, i.e. a plain [0, 1] image. Feeding it a
DINOv2-normalized image would apply that normalization twice.

Note [panovggt-outputs-local-points]:
``PanoVGGTModel.forward()`` returns ``local_points`` (camera frame), ``world_points`` /
``points`` (world frame), ``depth`` (distance along the ERP ray) and ``camera_poses``
(cam2world 4x4). We take the camera-frame points and the poses and re-derive the world
points ourselves, so that the pose/point relationship matches the rest of MapAnything
exactly. ``global_points`` is a separate head, not used here.

Note [panovggt-has-no-confidence]:
There is no confidence head, so ``conf`` is filled with ones to keep the unified output
format complete. Callers that filter on confidence will therefore keep everything.
"""

import torch
from omegaconf import OmegaConf
from panovggt.models.panovggt_model import PanoVGGTModel

from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.device import (
    get_amp_dtype,
    get_autocast_device_type,
    get_device,
)
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
)

# Architecture of the released checkpoint, mirroring the `model:` section of
# `PanoVGGT/training/config/default.yaml` (which is what `inference.py` and `app.py`
# build the model from). Overridable per-key via the `model_config` kwarg.
DEFAULT_MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": True,
    "enable_depth": True,
    "enable_point": True,
    "aggregator": {
        "depth": 36,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "patch_embed": "dinov2_vitl14_reg",
        "num_register_tokens": 5,
        "qkv_bias": True,
        "proj_bias": True,
        "ffn_bias": True,
        "qk_norm": True,
        "rope_freq": 100,
        "init_values": 0.01,
    },
}


def equirectangular_ray_directions(height, width, device, dtype):
    """
    Compute unit ray directions for an equirectangular (panoramic) image grid.

    Mirrors ``PanoVGGTModel._get_direction_vectors()`` exactly, including its half-pixel
    offset:
        phi   = ((x + 0.5) / W - 0.5) * 2 * pi
        theta = -((y + 0.5) / H - 0.5) * pi
        d = (cos(theta) sin(phi),  -sin(theta),  cos(theta) cos(phi))

    Note that the top row (y = 0) gives ``theta = +pi/2`` and hence ``d = (0, -1, 0)``:
    "up" is **-Y**, the same Y-down / Z-forward convention as the CasaMaestro wrapper.

    Args:
        height (int): Height of the equirectangular grid.
        width (int): Width of the equirectangular grid.
        device (torch.device): Device to create the directions on.
        dtype (torch.dtype): Dtype to create the directions with.

    Returns:
        torch.Tensor: Unit ray directions of shape (H, W, 3).
    """
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    phi = (u / width - 0.5) * 2 * torch.pi
    theta = -(v / height - 0.5) * torch.pi
    grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing="ij")

    cos_theta = torch.cos(grid_theta)
    ray_directions = torch.stack(
        [
            cos_theta * torch.sin(grid_phi),
            -torch.sin(grid_theta),
            cos_theta * torch.cos(grid_phi),
        ],
        dim=-1,
    )

    # Already unit norm analytically; normalize to guard against float error
    return ray_directions / ray_directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def cam2world_relative_to_first_view(cam2world):
    """
    Express cam2world poses relative to the first view, which becomes identity.

    PanoVGGT regresses each pose independently in its own learned frame, so the
    reconstruction is only defined up to a global rigid transform. Anchoring to the first
    view matches the MapAnything convention of using view 0 as the reference frame.

    Args:
        cam2world (torch.Tensor): cam2world poses of shape (B, V, 4, 4).

    Returns:
        torch.Tensor: Re-anchored cam2world poses of shape (B, V, 4, 4).
    """
    world2cam_first = torch.linalg.inv(cam2world[:, 0])
    relative = world2cam_first.unsqueeze(1) @ cam2world
    relative[:, 0] = torch.eye(4, device=cam2world.device, dtype=cam2world.dtype)

    return relative


class PanoVGGTWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        model_config=None,
        anchor_to_first_view=True,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        self.anchor_to_first_view = anchor_to_first_view

        # Architecture hyper-parameters, mirroring `training/config/default.yaml`
        config = dict(DEFAULT_MODEL_CONFIG)
        config["aggregator"] = dict(DEFAULT_MODEL_CONFIG["aggregator"])
        if model_config is not None:
            overrides = OmegaConf.to_container(
                OmegaConf.create(model_config), resolve=True
            )
            config["aggregator"].update(overrides.pop("aggregator", {}))
            config.update(overrides)

        self.model = PanoVGGTModel(**config)

        # Load the released PanoVGGT weights
        print(f"Loading PanoVGGT checkpoint from {self.ckpt_path} ...")
        state_dict = self._load_state_dict(self.ckpt_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[PanoVGGT load] missing={len(missing)} unexpected={len(unexpected)}")

        self.device = get_device()
        # Get the dtype for PanoVGGT inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = get_amp_dtype(self.device)

    @staticmethod
    def _load_state_dict(ckpt_path):
        """
        Load a PanoVGGT state dict, following `load_model()` in the reference repo.

        Args:
            ckpt_path (str): Path to the checkpoint.

        Returns:
            dict: The model state dict.
        """
        if str(ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            for key in ("model_state_dict", "model", "state_dict"):
                if isinstance(state_dict, dict) and key in state_dict:
                    state_dict = state_dict[key]
                    break

        # Strip the DistributedDataParallel prefix
        return {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

    def forward(self, views):
        """
        Forward pass wrapper for PanoVGGT

        Assumptions:
        - All the input views have the same image shape.
        - The input images are equirectangular panoramas.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        _, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # PanoVGGT normalizes the images inside its own aggregator
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "PanoVGGT normalizes internally, so it expects an un-normalized image"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Run the PanoVGGT model
        with torch.autocast(get_autocast_device_type(self.device), dtype=self.dtype):
            results = self.model(images)

        # Need high precision for transformations
        with torch.autocast(get_autocast_device_type(self.device), enabled=False):
            # PanoVGGT's camera head already outputs cam2world
            pred_cam2world = results["camera_poses"].float()
            if self.anchor_to_first_view:
                pred_cam2world = cam2world_relative_to_first_view(pred_cam2world)

            # Analytic equirectangular ray directions, shared across all views
            ray_directions = equirectangular_ray_directions(
                height, width, images.device, torch.float32
            )

            res = []
            for view_idx in range(num_views):
                # PanoVGGT predicts distance along the ERP viewing ray
                curr_view_depth_along_ray = results["depth"][:, view_idx, ...].float()
                curr_view_pts3d_cam = results["local_points"][
                    :, view_idx, ...
                ].float()

                # Expand the ray directions to the batch size of the view
                curr_view_ray_dirs = ray_directions.expand(
                    curr_view_depth_along_ray.shape[0], -1, -1, -1
                )

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

                # PanoVGGT has no confidence head; see
                # note [panovggt-has-no-confidence]
                curr_view_confidence = torch.ones_like(
                    curr_view_depth_along_ray[..., 0]
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
