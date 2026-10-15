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

"""Project-wide constants for Xiaomi-Lens-0."""

# Keep generation defaults aligned across inference entry points.
DEFAULT_MAX_NEW_TOKENS = 16_384

MASK_TOKEN = "<|mask|>"

# Grounding delimiters. The model emits, per instance,
# `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`. The bbox delimiters
# reuse native Qwen token IDs 248049/248050; the mask delimiters are ordinary
# text. Only the `<|mask|>` slot is an added tokenizer row.
BOX_START_TOKEN = "<bbox>"
BOX_END_TOKEN = "</bbox>"
MASK_START_TOKEN = "<mask>"
MASK_END_TOKEN = "</mask>"
