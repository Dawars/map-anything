# Copyright (c) 2026 AutoLab, Shanghai Jiao Tong University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Panoramic camera decoder for CasaMaestro.

Vendored verbatim from the CasaMaestro release
(``src/depth_anything_3/model/cam_pano.py``).

CasaMaestro is built on top of Depth Anything 3 and its ``depth_anything_3``
package is byte-identical to the Depth Anything 3 fork already vendored via the
``depth-anything-3`` optional dependency, with the sole exception of this module
(and the training-only ``train_pano`` package). Vendoring just this file lets
CasaMaestro and Depth Anything 3 coexist without a duplicate ``depth_anything_3``
package on the ``PYTHONPATH``.
"""

import torch
import torch.nn as nn


def safe_normalize_quat(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # q: (...,4)
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))


class CameraDecPano(nn.Module):
    """
    pano camera decoder:
      - per-view MLP with LN
      - cross-view self-attention
      - quaternion normalized to unit length
      - fov is fixed constant
    """

    def __init__(
        self,
        dim_in: int = 1536,
        dim_model: int = 512,
        depth: int = 2,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        fixed_fov=(0.0, 0.0),
        learn_fov: bool = False,
    ):
        super().__init__()
        self.fixed_fov = torch.tensor(fixed_fov, dtype=torch.float32).view(1, 1, 2)
        self.learn_fov = learn_fov
        if learn_fov:
            self.fov_param = nn.Parameter(self.fixed_fov.clone())

        # Project to transformer dim
        self.proj = nn.Sequential(
            nn.Linear(dim_in, dim_model),
            nn.LayerNorm(dim_model),
        )

        # Per-view refinement MLP (pre-attention)
        hidden = int(dim_model * mlp_ratio)
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_model),
            nn.Dropout(dropout),
        )
        self.pre_ln = nn.LayerNorm(dim_model)

        # Cross-view self-attention
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Output heads
        self.head_ln = nn.LayerNorm(dim_model)
        self.fc_t = nn.Linear(dim_model, 3)
        self.fc_q = nn.Linear(dim_model, 4)

        # small scale on rotation at init
        nn.init.zeros_(self.fc_t.bias)
        nn.init.zeros_(self.fc_q.bias)

    def forward(self, feat, camera_encoding=None, attn_mask=None, *args, **kwargs):
        """
        feat: (B, V, C)
        attn_mask: optional transformer attn mask for views
        """
        B, V, _ = feat.shape

        x = self.proj(feat)  # (B,V,dim_model)
        x = self.pre_ln(x + self.pre_mlp(x))
        x = self.view_encoder(x, mask=attn_mask)  # (B,V,dim_model)
        x = self.head_ln(x)

        out_t = self.fc_t(x)  # (B,V,3)

        if camera_encoding is None:
            out_q = safe_normalize_quat(self.fc_q(x))  # (B,V,4)
        else:
            # original da3 behavior
            out_q = safe_normalize_quat(camera_encoding[..., 3:7])

        # fixed fov
        if self.learn_fov:
            out_fov = self.fov_param.expand(B, V, 2)
        else:
            out_fov = self.fixed_fov.to(x.device).expand(B, V, 2)

        pose_enc = torch.cat([out_t, out_q, out_fov], dim=-1)  # (B,V,9)
        return pose_enc
