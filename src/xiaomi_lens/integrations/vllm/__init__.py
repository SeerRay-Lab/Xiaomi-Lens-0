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

"""Xiaomi-Lens-0 integrations with vLLM for text generation and mask decoding.

Serving uses two vLLM processes: `run_vllm.sh` generates text, and
`run_vllm_mask.sh` decodes mask tokens through the pooling API. `vllm_model`
extends vLLM's Qwen3.5 model with PixelMaskDecoder; `io_processor` handles
requests and serializes masks. Both plugins are registered by `pyproject.toml`.

`client` sends the image and instruction to the text server, then replays the
completion with the image at the pooling server. `prepare_served` prepares the
checkpoint views used by the two launchers.

`mask_service` provides a separate Hugging Face fallback at `/decode_mask`,
using `infer.run_forward`. The pooling client uses `/pooling` instead.
"""
