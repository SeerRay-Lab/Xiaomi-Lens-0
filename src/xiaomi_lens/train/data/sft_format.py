# Copyright 2026 Xiaomi Corporation.
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

"""Build assistant completions in the Xiaomi-Lens-0 grounding format.

Each instance is serialized as::

    <bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>

- A bbox is an integer per-mille [0, 1000] quadruple serialized with
  ``json.dumps(bbox)``, for example ``[534, 320, 856, 995]``.
- Multi-instance completions concatenate blocks directly, in the same order
  as the sample's ``segmentation`` list.
- Localization without a mask emits only ``<bbox>[...]</bbox>``.
"""

import json

from ...constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    MASK_END_TOKEN,
    MASK_START_TOKEN,
    MASK_TOKEN,
)


def box_token(bbox) -> str:
    """``<bbox>[x1, y1, x2, y2]</bbox>`` (per-mille bbox)."""
    return f"{BOX_START_TOKEN}{json.dumps(bbox)}{BOX_END_TOKEN}"


def mask_block(bbox) -> str:
    """Build one box-and-mask instance block."""
    return f"{box_token(bbox)}{MASK_START_TOKEN}{MASK_TOKEN}{MASK_END_TOKEN}"


def build_completion(bboxes, *, with_mask: bool = True) -> str:
    """Concatenate per-instance blocks into an assistant completion.

    Args:
        bboxes: list[list[int]] of per-mille boxes in completion order, matching
            the segmentation slot order.
        with_mask: Emit box-and-mask blocks when True, or box-only blocks for
            localization without masks when False.
    """
    blk = mask_block if with_mask else box_token
    return "".join(blk(b) for b in bboxes)
