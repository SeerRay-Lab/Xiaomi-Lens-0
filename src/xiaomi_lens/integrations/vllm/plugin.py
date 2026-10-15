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

"""vLLM general-plugin entry point for Xiaomi-Lens-0.

Registered via `pyproject.toml` under `[project.entry-points."vllm.general_plugins"]`.
vLLM calls `load_general_plugins()` in every process it spawns — the API server AND
each EngineCore worker — so registering the custom model here (rather than with a
one-off `ModelRegistry.register_model(...)` in the launcher) is what makes it visible
across the v1 multiprocess boundary.

The model class is passed as a `"module:Class"` string so vLLM imports it lazily,
inside the worker, only when the architecture is actually requested — keeping this
entry point cheap to load.
"""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    # Map the checkpoint's config.json architecture to our vLLM model that loads
    # PixelMaskDecoder and exposes a mask-producing pooler. No --hf-overrides
    # needed once this is registered (vLLM matches the arch name directly).
    ModelRegistry.register_model(
        "XiaomiLensQwen3_5ForConditionalGeneration",
        "xiaomi_lens.integrations.vllm.vllm_model:XiaomiLensForVLLM",
    )
