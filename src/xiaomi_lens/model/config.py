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

"""Model-configuration helpers for Xiaomi-Lens-0."""


def get_qwen_vision_config(config):
    for attr_name in ("vision_config", "visual_config"):
        vision_config = getattr(config, attr_name, None)
        if vision_config is not None:
            return vision_config
    raise ValueError("Qwen vision config (vision_config/visual_config) is required to build PixelMaskDecoder.")


def build_mask_decoder_config(config) -> dict:
    vision_config = get_qwen_vision_config(config)
    text_config = config.text_config
    # Project pre-merge vision features to the LLM width for patch attention,
    # then reduce channels before the decoder's transposed convolutions.
    return {
        "llm_hidden_size": text_config.hidden_size,
        "llm_intermediate_size": text_config.intermediate_size,
        "llm_num_heads": text_config.num_attention_heads,
        "vision_hidden_size": vision_config.hidden_size,
        "mask_decoder_channels": int(getattr(config, "mask_decoder_channels", 256)),
        "mask_decoder_embed_dim": int(getattr(config, "mask_decoder_embed_dim", 32)),
    }
