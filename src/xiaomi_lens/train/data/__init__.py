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

"""Lazy JSONL data pipeline: manifests, chat rendering, and per-sample encoding."""

from .io import (
    DatasetEntry,
    load_dataset_config,
    parse_path_spec,
    resolve_dataset_entries,
    resolve_sample_images,
)


# Load modules that import numpy-heavy or processor code on demand.
def __getattr__(name):
    if name == "encode_sample":
        from .preprocess import encode_sample
        return encode_sample
    elif name == "JsonlIndexDataset":
        from .jsonl_index import JsonlIndexDataset
        return JsonlIndexDataset
    elif name == "build_jsonl_index_dataset":
        from .jsonl_index import build_jsonl_index_dataset
        return build_jsonl_index_dataset
    elif name == "decode_coco_rle":
        from .segmentation import decode_coco_rle
        return decode_coco_rle
    elif name == "segmentation_to_pixel_mask":
        from .segmentation import segmentation_to_pixel_mask
        return segmentation_to_pixel_mask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetEntry",
    "build_jsonl_index_dataset",
    "decode_coco_rle",
    "encode_sample",
    "JsonlIndexDataset",
    "load_dataset_config",
    "parse_path_spec",
    "resolve_dataset_entries",
    "resolve_sample_images",
    "segmentation_to_pixel_mask",
]
