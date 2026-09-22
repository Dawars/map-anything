# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Anchor3R

Anchor3R (https://github.com/polar-explorer/Anchor3R, arXiv:2606.05035) ships as a
standalone repository rather than a pip package, so this wrapper makes it
importable (see `_ensure_anchor3r_importable`) and adapts its windowed streaming
forward pass to MapAnything's unified `forward(views) -> List[dict]` model
interface, the same way `mapanything/models/external/horizonstream` does.

Anchor3R is architecturally a follow-up to HorizonStream from an overlapping
author group ("Streaming 3D Reconstruction with Transient Anchors for
Long-Horizon Visual Mapping"): it processes a sequence in bounded windows with
KV-cache carry-over rather than attending over the whole sequence at once, so
this wrapper chunks the input views the same way
`anchor3r/runtime/inference.py::run_inference` does, and stitches the per-chunk
camera predictions with Anchor3R's own `build_camera_trajectories` (a proper
sparse least-squares rotation/position averaging over the overlapping
per-window pose estimates, solved with `scikit-sparse`/CHOLMOD - closer to a
real pose-graph optimization than HorizonStream's median-based motion
averaging, though still not photometric bundle adjustment).

Unlike HorizonStream's wrapper, no manual precision-split re-implementation of
the model's forward pass is needed here: Anchor3R's own `MultiViewStreamSampler.
forward()` already disables autocast around the camera head, FOV head, and
depth head (see its `decoder_autocast` block), so wrapping the aggregator call
in mixed precision is enough - `Anchor3RModel.forward_chunk` already gives
accurate, full-precision poses/depth for free.

Hard dependency note: `anchor3r.geometry.camera` (needed for
`build_camera_trajectories`) imports `sksparse.cholmod` at module level, i.e.
`scikit-sparse` + the native SuiteSparse/CHOLMOD library are required to import
this wrapper at all, not just for the offline trajectory.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

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

# Released checkpoint (see Anchor3R/checkpoints/release.json)
_DEFAULT_HF_REPO_ID = "polar-explorer/Anchor3R"
_DEFAULT_HF_FILENAME = "Anchor3R.pt"
_DEFAULT_HF_REVISION = "aae563ce2831699554b5f83bf1c5e31775959d49"


def _ensure_anchor3r_importable():
    """
    Make the `anchor3r` package importable.

    Anchor3R is not published as a pip package, so this looks for (in order):
    1. An already-importable `anchor3r` (e.g. installed with `pip install -e`,
       or already on PYTHONPATH).
    2. `ANCHOR3R_REPO_PATH` env var pointing at the Anchor3R repo root.
    3. A sibling `Anchor3R` checkout next to the `map-anything` repo.
    """
    try:
        import anchor3r  # noqa: F401

        return
    except ImportError:
        pass

    candidates = []
    env_path = os.environ.get("ANCHOR3R_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # This file lives at:
    #   <root>/map-anything/mapanything/models/external/anchor3r/__init__.py
    # so five levels up from this file is <root>, the directory containing both
    # map-anything and Anchor3R as sibling checkouts.
    this_file = Path(__file__).resolve()
    if len(this_file.parents) > 5:
        candidates.append(this_file.parents[5] / "Anchor3R")

    for candidate in candidates:
        if (candidate / "anchor3r" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            import anchor3r  # noqa: F401

            return

    raise ImportError(
        "Could not import the 'anchor3r' package. Install it "
        "(`pip install -e /path/to/Anchor3R --no-deps`), put it on PYTHONPATH, or "
        "set the ANCHOR3R_REPO_PATH environment variable to the Anchor3R repo root."
    )


def _resolve_checkpoint_path(
    checkpoint, hf_repo_id, hf_filename, hf_revision, hf_local_dir
):
    if checkpoint:
        return checkpoint
    if not hf_repo_id or not hf_filename:
        return None
    from huggingface_hub import hf_hub_download

    os.makedirs(hf_local_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_filename,
        revision=hf_revision,
        local_dir=hf_local_dir,
    )


def _chunk_schedule(
    num_frames: int, window_size: int, chunk_size: int
) -> List[Tuple[int, int]]:
    """
    Same schedule as anchor3r/runtime/inference.py::_chunk_ranges: the first
    chunk covers `window_size` frames, subsequent chunks slide forward by
    `chunk_size`.
    """
    if num_frames <= 0:
        return []
    if num_frames <= window_size:
        return [(0, num_frames)]
    chunks = [(0, window_size)]
    start = window_size
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        chunks.append((start, end))
        start = end
    return chunks


class Anchor3RWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        checkpoint=None,
        hf_repo_id=_DEFAULT_HF_REPO_ID,
        hf_filename=_DEFAULT_HF_FILENAME,
        hf_revision=_DEFAULT_HF_REVISION,
        hf_local_dir="checkpoints",
        strict_load=True,
        sampler_cfg=None,
        # Streaming-window controls (see module docstring). Defaults match
        # anchor3r/configs/infer.yaml, the released checkpoint's own
        # recommended inference settings.
        window_size=10,
        chunk_size=64,
        offload_outputs_to_cpu=False,
        # "offline" runs Anchor3R's sparse least-squares pose-graph averaging
        # over all overlapping window estimates (better for a completed batch
        # reconstruction); "online" is the causal, real-time-compatible
        # trajectory Anchor3R's own reference config defaults to.
        mode="offline",
    ):
        super().__init__()
        _ensure_anchor3r_importable()

        from anchor3r.runtime.model import Anchor3RModel

        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload

        resolved_checkpoint = None
        if load_pretrained_weights:
            resolved_checkpoint = _resolve_checkpoint_path(
                checkpoint, hf_repo_id, hf_filename, hf_revision, hf_local_dir
            )

        self.model = Anchor3RModel(
            sampler_cfg or {},
            checkpoint=resolved_checkpoint,
            strict_load=strict_load,
        )

        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if mode not in ("online", "offline"):
            raise ValueError(f"mode must be 'online' or 'offline', got: {mode!r}")
        self.window_size = int(window_size)
        self.chunk_size = int(chunk_size)
        self.offload_outputs_to_cpu = bool(offload_outputs_to_cpu)
        self.mode = mode

        self.device = get_device()
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = get_amp_dtype(self.device)

    def forward(self, views):
        """
        Forward pass wrapper for Anchor3R.

        Assumptions:
        - All the input views have the same image shape.
        - Batch size per view is 1 (Anchor3R's streaming inference only
          supports a single sequence at a time).
        - Views are processed in temporal order as a single streaming
          sequence, internally split into windows of `self.window_size` /
          `self.chunk_size` frames with KV-cache carry-over between them (see
          module docstring for why this chunking is required).

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)
        assert batch_size_per_view == 1, (
            "Anchor3R only supports a batch size of 1 per view (a single sequence at a time)."
        )

        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "Anchor3R expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor in [0, 1]
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        device_type = get_autocast_device_type(self.device)
        use_amp = device_type == "cuda"  # matches Anchor3R's own reference pipeline
        state = self.model.new_state()
        chunks = _chunk_schedule(num_views, self.window_size, self.chunk_size)

        chunk_cam_maps = []
        fov_chunks = []
        dpt_map_chunks = []
        dpt_cnf_chunks = []
        for chunk_idx, (start, end) in enumerate(chunks):
            chunk_images = images[:, start:end]
            current_window = end - start if chunk_idx == 0 else self.window_size

            with torch.autocast(device_type, enabled=use_amp, dtype=self.dtype):
                out = self.model.forward_chunk(
                    chunk_images,
                    window_size=current_window,
                    chunk_idx=chunk_idx,
                    state=state,
                )

            cam = out["chunk_cam_map"].detach().float()
            fov = out["fov"].detach().float()
            dpt_map = out["dpt_map"].detach().float()
            dpt_cnf = out["dpt_cnf"].detach().float()
            if self.offload_outputs_to_cpu:
                cam = cam.cpu()
                fov = fov.cpu()
                dpt_map = dpt_map.cpu()
                dpt_cnf = dpt_cnf.cpu()
            chunk_cam_maps.append(cam)
            fov_chunks.append(fov)
            dpt_map_chunks.append(dpt_map)
            dpt_cnf_chunks.append(dpt_cnf)

            self.model.advance_state(state, is_last=(chunk_idx == len(chunks) - 1))

        from anchor3r.geometry.camera import build_camera_trajectories, decode_cam_map

        # Stitch the per-chunk (possibly overlapping) pose estimates into one
        # consistent absolute trajectory for the whole sequence.
        trajectory_maps = build_camera_trajectories(
            chunk_cam_maps,
            fov_chunks,
            num_frames=num_views,
            window_size=min(self.window_size, num_views),
        )
        cam_map = trajectory_maps[self.mode].to(device=self.device, dtype=torch.float32)
        w2c, intri = decode_cam_map(cam_map, (height, width))  # (B, V, 3, 4), (B, V, 3, 3)

        dpt_map = torch.cat(dpt_map_chunks, dim=1).to(self.device)  # (B, V, H*W, 1)
        dpt_cnf = torch.cat(dpt_cnf_chunks, dim=1).to(self.device)  # (B, V, H*W, 1)
        if dpt_map.shape[1] != num_views:
            raise RuntimeError(
                f"Expected {num_views} depth frames from streaming inference, got {dpt_map.shape[1]}"
            )
        depth = dpt_map.reshape(1, num_views, height, width)  # (B, V, H, W)
        depth_conf = dpt_cnf.reshape(1, num_views, height, width)  # (B, V, H, W)

        # Convert the output to MapAnything format
        res = []
        for view_idx in range(num_views):
            # Get the extrinsics
            curr_view_extrinsic = w2c[:, view_idx, ...]
            curr_view_extrinsic = closed_form_inverse_se3(
                curr_view_extrinsic
            )  # Convert to cam2world
            curr_view_intrinsic = intri[:, view_idx, ...]
            curr_view_depth_z = depth[:, view_idx, ...]
            curr_view_confidence = depth_conf[:, view_idx, ...]

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
