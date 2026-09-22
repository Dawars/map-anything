# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for HorizonStream

HorizonStream (https://github.com/3DAgentWorld/HorizonStream) ships as a standalone
repository rather than a pip package, so this wrapper makes it importable (see
`_ensure_horizonstream_importable`) and adapts its windowed streaming forward pass
to MapAnything's unified `forward(views) -> List[dict]` model interface.

HorizonStream is architecturally a *streaming* model: it processes a sequence in
bounded windows with KV-cache carry-over rather than attending over the whole
sequence at once (its first-chunk attention mask is O(window_size^2), so feeding
it hundreds of views as a single window will blow up memory). This wrapper
therefore internally chunks the input views the same way
`horizonstream/core/infer.py::run_inference_cfg` does, and stitches the per-chunk
camera predictions back into one consistent trajectory with HorizonStream's own
`compute_motion_averaged_camera_maps` utility.
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


def _ensure_horizonstream_importable():
    """
    Make the `horizonstream` package importable.

    HorizonStream is not published as a pip package, so this looks for (in order):
    1. An already-importable `horizonstream` (e.g. installed with `pip install -e`,
       or already on PYTHONPATH).
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
        "(`pip install -e /path/to/horizon-stream`), put it on PYTHONPATH, or set "
        "the HORIZONSTREAM_REPO_PATH environment variable to the horizon-stream repo root."
    )


def _chunk_schedule(
    num_frames: int, window_size: int, sliding_size: int
) -> List[Tuple[int, int]]:
    """
    Same schedule as horizonstream/core/infer.py::_chunk_schedule: the first chunk
    covers `window_size` frames, subsequent chunks slide forward by `sliding_size`.
    """
    if num_frames <= 0:
        return []
    if num_frames <= window_size:
        return [(0, num_frames)]
    chunks = [(0, window_size)]
    start = window_size
    while start < num_frames:
        end = min(start + sliding_size, num_frames)
        chunks.append((start, end))
        start = end
    return chunks


def _forward_chunk_high_precision(hs, images, *, window_size, chunk_idx, state, device_type, amp_dtype):
    """
    Re-implementation of `HorizonStream.forward_chunk`
    (horizon-stream/horizonstream/models/horizonstream.py) that keeps the token
    aggregator in mixed precision but forces the camera head, depth head, and
    metric-scale readout to run in full precision (fp32), for accurate final
    poses / depth. Mirrors the upstream implementation line-for-line other than
    the added precision split.
    """
    B, S, _, _, _ = images.shape
    Win = int(window_size.item()) if hasattr(window_size, "item") else int(window_size)

    with torch.autocast(device_type, dtype=amp_dtype):
        output_dict, win_pose_tokens, patch_start_idx = hs.agg_regator(
            images,
            frame_kv_caches=state["frame_kv_caches"],
            global_kv_caches=state["global_kv_caches"],
            n_views=1,
            window_size=Win,
            chunk_idx=chunk_idx,
            rope_frame_start=(
                (chunk_idx * S) % hs.rope_temporal_period
                if hs.rope_temporal_period > 0
                else 0
            ),
            gla_cache=state.get("gla_cache"),
        )

    with torch.autocast(device_type, enabled=False):
        # `_output_dict_to_list` returns a list indexed by absolute layer number,
        # padded with `None` at layers that aren't required by the DPT head /
        # metric-scale readout (only specific intermediate_layer_idx positions
        # are populated) - only cast the populated entries.
        aggregated_tokens_list = [
            tok.float() if tok is not None else None
            for tok in hs._output_dict_to_list(output_dict)
        ]
        chunk_cam_maps_raw = hs.cam_decoder(win_pose_tokens.float(), chunk_idx=chunk_idx)

        predicted_metric_scale, _ = hs._predict_metric_scale(
            aggregated_tokens_list, patch_start_idx
        )
        chunk_cam_maps = hs._scale_chunk_camera_maps(
            chunk_cam_maps_raw, predicted_metric_scale, B, S
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
            ).view(B, 1, 1, 1, 1)
            depth = depth * scale

    return {
        "chunk_cam_map": chunk_cam_maps[-1],
        "depth": depth,
        "depth_conf": depth_conf,
    }


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
        # Streaming-window controls (see module docstring). Defaults match
        # horizon-stream/configs/horizonstream_infer.yaml, the released
        # checkpoint's own recommended inference settings. Lower these (e.g.
        # window_size=6-8) to trade a little accuracy/speed for less peak VRAM.
        window_size=10,
        sliding_size=21,
        offload_outputs_to_cpu=False,
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

        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if sliding_size <= 0:
            raise ValueError("sliding_size must be positive.")
        self.window_size = int(window_size)
        self.sliding_size = int(sliding_size)
        self.offload_outputs_to_cpu = bool(offload_outputs_to_cpu)

        self.device = get_device()
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = get_amp_dtype(self.device)

    def forward(self, views):
        """
        Forward pass wrapper for HorizonStream.

        Assumptions:
        - All the input views have the same image shape.
        - Batch size per view is 1 (HorizonStream's streaming inference only
          supports a single sequence at a time).
        - Views are processed in temporal order as a single streaming sequence,
          internally split into windows of `self.window_size` /
          `self.sliding_size` frames with KV-cache carry-over between them (see
          module docstring for why this chunking is required).

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
        assert batch_size_per_view == 1, (
            "HorizonStream only supports a batch size of 1 per view (a single sequence at a time)."
        )

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
        device_type = get_autocast_device_type(self.device)
        state = hs.build_sequence_state()
        chunks = _chunk_schedule(num_views, self.window_size, self.sliding_size)

        chunk_cam_maps = []
        global_depth = []
        global_depth_conf = []
        for chunk_idx, (start, end) in enumerate(chunks):
            chunk_images = images[:, start:end]
            current_window_size = end - start if chunk_idx == 0 else self.window_size

            outputs = _forward_chunk_high_precision(
                hs,
                chunk_images,
                window_size=current_window_size,
                chunk_idx=chunk_idx,
                state=state,
                device_type=device_type,
                amp_dtype=self.dtype,
            )

            chunk_cam_map = outputs["chunk_cam_map"].detach().float()
            depth_chunk = outputs["depth"].float()
            depth_conf_chunk = outputs["depth_conf"].float()
            if self.offload_outputs_to_cpu:
                chunk_cam_map = chunk_cam_map.cpu()
                depth_chunk = depth_chunk.cpu()
                depth_conf_chunk = depth_conf_chunk.cpu()
            chunk_cam_maps.append(chunk_cam_map)
            for frame_idx in range(depth_chunk.shape[1]):
                global_depth.append(depth_chunk[:, frame_idx])
                global_depth_conf.append(depth_conf_chunk[:, frame_idx])

            hs.advance_sequence_state(state, is_last_chunk=(chunk_idx == len(chunks) - 1))

        if len(global_depth) != num_views:
            raise RuntimeError(
                f"Expected {num_views} depth frames from streaming inference, got {len(global_depth)}"
            )

        from horizonstream.runtime.motion_averaging import (
            compute_motion_averaged_camera_maps,
        )
        from horizonstream.utils.vendor.models.components.utils.pose_enc import (
            pose_encoding_to_extri_intri,
        )

        # Stitch the per-chunk (possibly overlapping) pose estimates into one
        # consistent absolute trajectory for the whole sequence.
        motion_maps = compute_motion_averaged_camera_maps(
            chunk_cam_maps,
            frames_num=num_views,
            window_size=self.window_size,
            dtype=torch.float32,
            enable_offline=False,
        )
        pose_enc = motion_maps["online_cam_map"].to(device=self.device, dtype=torch.float32)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (height, width))

        depth = torch.stack(global_depth, dim=1).to(self.device)  # (B, V, H, W, 1)
        depth_conf = torch.stack(global_depth_conf, dim=1).to(self.device)  # (B, V, H, W)

        # Convert the output to MapAnything format
        res = []
        for view_idx in range(num_views):
            # Get the extrinsics
            curr_view_extrinsic = extrinsic[:, view_idx, ...]
            curr_view_extrinsic = closed_form_inverse_se3(
                curr_view_extrinsic
            )  # Convert to cam2world
            curr_view_intrinsic = intrinsic[:, view_idx, ...]
            curr_view_depth_z = depth[:, view_idx, ...].squeeze(-1)
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
