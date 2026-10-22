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

"""Training for Xiaomi-Lens-0. Install with `pip install -e ".[train]"`."""


# Load torch / transformers only when a trainer symbol is requested.
def __getattr__(name):
    if name == "XiaomiLensTrainer":
        from .trainer import XiaomiLensTrainer
        return XiaomiLensTrainer
    elif name == "XiaomiLensModelConfig":
        from .config import XiaomiLensModelConfig
        return XiaomiLensModelConfig
    elif name == "XiaomiLensScriptArguments":
        from .config import XiaomiLensScriptArguments
        return XiaomiLensScriptArguments
    elif name == "XiaomiLensTrainingConfig":
        from .config import XiaomiLensTrainingConfig
        return XiaomiLensTrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "XiaomiLensModelConfig",
    "XiaomiLensScriptArguments",
    "XiaomiLensTrainer",
    "XiaomiLensTrainingConfig",
]
