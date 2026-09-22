# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for HorizonStream

HorizonStream (https://github.com/3DAgentWorld/HorizonStream) ships as a standalone
repository rather than a pip package, so this wrapper makes it importable (see
`_ensure_horizonstream_importable`) and adapts its windowed streaming forward pass
to MapAnything's unified `forward(views) -> List[dict]` model interface.
"""

import os
import sys
from pathlib import Path

import torch

from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.device import get_amp_dtype, get_autocast_device_type, get_device
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


def _ensure_horizonstream_importable():
    """
    Make the `horizonstream` package importable.

    HorizonStream is not published as a pip package, so this looks for (in order):
    1. An already-importable `horizonstream` (e.g. installed with `pip install -e`).
    2. `HORIZONSTREAM_REPO_PATH` env var pointing at the horizon-stream repo root.
    3. A sibling `horizon-stream` checkout next to the `map-anything` repo.
    """
    try:
        import horizonstream  # noqa: F401

        return
    except ImportError:
        pass

    candidates = []
    env_path = os.environ.get("HORIZONSTREAM_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # This file lives at:
    #   <root>/map-anything/mapanything/models/external/horizonstream/__init__.py
    # so five levels up from this file is <root>, the directory containing both
    # map-anything and horizon-stream as sibling checkouts.
    this_file = Path(__file__).resolve()
    if len(this_file.parents) > 5:
        candidates.append(this_file.parents[5] / "horizon-stream")

    for candidate in candidates:
        if (candidate / "horizonstream" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            import horizonstream  # noqa: F401

            return

    raise ImportError(
        "Could not import the 'horizonstream' package. Install it "
        "(`pip install -e /path/to/horizon-stream`) or set the HORIZONSTREAM_REPO_PATH "
        "environment variable to the horizon-stream repo root."
    )


class HorizonStreamWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        checkpoint=None,
        hf_repo_id="NicolasCC/HorizonStream",
        hf_filename="HorizonStream.pt",
        hf_local_dir="checkpoints",
        strict_load=True,
        horizonstream_cfg=None,
    ):
        super().__init__()
        _ensure_horizonstream_importable()

        from horizonstream.core.model import HorizonStreamModel

        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload

        model_cfg = {
            "checkpoint": checkpoint,
            "strict_load": strict_load,
            "horizonstream_cfg": dict(horizonstream_cfg or {}),
        }
        if load_pretrained_weights:
            if checkpoint:
                model_cfg["hf"] = None
            else:
                model_cfg["hf"] = {
                    "repo_id": hf_repo_id,
                    "filename": hf_filename,
                    "local_dir": hf_local_dir,
                }
        else:
            model_cfg["checkpoint"] = None
            model_cfg["hf"] = None

        self.model = HorizonStreamModel(model_cfg)

        self.device = get_device()
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = get_amp_dtype(self.device)

    def forward(self, views):
        """
        Forward pass wrapper for HorizonStream.

        Assumptions:
        - All the input views have the same image shape.
        - The full view sequence is run as a single window (chunk_idx=0), i.e. no
          streaming chunking / KV-cache carry-over across separate calls. This
          matches HorizonStream's own `forward_window` single-shot entry point.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # HorizonStream expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "HorizonStream expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor in [0, 1]
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        hs = self.model.horizonstream
        state = hs.build_sequence_state()
        device_type = get_autocast_device_type(self.device)

        # Run the token aggregator in mixed precision
        with torch.autocast(device_type, dtype=self.dtype):
            output_dict, win_pose_tokens, patch_start_idx = hs.agg_regator(
                images,
                frame_kv_caches=state["frame_kv_caches"],
                global_kv_caches=state["global_kv_caches"],
                n_views=1,
                window_size=num_views,
                chunk_idx=0,
                rope_frame_start=0,
                gla_cache=state.get("gla_cache"),
            )

        # Run the camera head, depth head, and all geometric post-processing in
        # full precision (fp32) for accurate final poses and point maps
        with torch.autocast(device_type, enabled=False):
            aggregated_tokens_list = [
                tok.float() for tok in hs._output_dict_to_list(output_dict)
            ]

            chunk_cam_maps_raw = hs.cam_decoder(win_pose_tokens.float(), chunk_idx=0)
            predicted_metric_scale, _ = hs._predict_metric_scale(
                aggregated_tokens_list, patch_start_idx
            )
            chunk_cam_maps = hs._scale_chunk_camera_maps(
                chunk_cam_maps_raw,
                predicted_metric_scale,
                batch_size_per_view,
                num_views,
            )

            depth, depth_conf = hs.dpt_decoder(
                aggregated_tokens_list,
                images=images.float(),
                patch_start_idx=patch_start_idx,
                frames_chunk_size=hs.frames_chunk_size,
            )
            if predicted_metric_scale is not None:
                scale = predicted_metric_scale.to(
                    dtype=depth.dtype, device=depth.device
                ).view(batch_size_per_view, 1, 1, 1, 1)
                depth = depth * scale

            # chunk_cam_maps[-1] is the last refinement iteration's pose encoding,
            # shape (B * num_views, num_views, cam_dim). Since chunk_idx=0 and
            # window_size == num_views, the row at index [num_views - 1] holds the
            # full-context (non-causal) pose estimate for every view.
            cam_dim = chunk_cam_maps[-1].shape[-1]
            pose_enc = chunk_cam_maps[-1].reshape(
                batch_size_per_view, -1, num_views, cam_dim
            )[:, -1, :, :9].float()

            from horizonstream.utils.vendor.models.components.utils.pose_enc import (
                pose_encoding_to_extri_intri,
            )

            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, (height, width)
            )

            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics
                curr_view_extrinsic = extrinsic[:, view_idx, ...]
                curr_view_extrinsic = closed_form_inverse_se3(
                    curr_view_extrinsic
                )  # Convert to cam2world
                curr_view_intrinsic = intrinsic[:, view_idx, ...]
                curr_view_depth_z = depth[:, view_idx, ...].float().squeeze(-1)
                curr_view_confidence = depth_conf[:, view_idx, ...].float()

                # Get the camera frame pointmaps
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the extrinsics to quaternions and translations
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

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
