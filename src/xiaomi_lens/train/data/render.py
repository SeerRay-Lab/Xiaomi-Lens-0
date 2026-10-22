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

"""Chat-template rendering helpers for lazy preprocessing.

Pure-Python; importable without torch, transformers, or datasets.
"""

from __future__ import annotations

import logging
from typing import Any

import PIL.Image

logger = logging.getLogger(__name__)


def _count_image_placeholders(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            total += m["content"].count("<image>")
    return total


def ensure_user_message(
    messages: list[dict],
    num_images: int,
    *,
    sample_id: str | None = None,
) -> list[dict]:
    """Synthesize a leading user message when a sample has none.

    Stage-1 (pretrain) data may ship as a bare assistant turn:

        {"messages": [{"role": "assistant", "content": "..."}],
         "images": [...]}

    Two downstream paths break without a user turn:

    1. `Processor.apply_chat_template([], add_generation_prompt=True)` in
       `render_prompt_and_completion` raises `IndexError` on empty input
       for Qwen3.5 templates. A trivial `[{"role": "user", "content": ""}]`
       renders cleanly.
    2. `align_image_placeholders` only counts `<image>` in user messages,
       so samples with `num_images>0` and no user turn fall through to
       a warning and hit the Qwen processor's "tokens: 0, features: N"
       mismatch.

    Repair policy:
    - No user message exists → insert one at the front.
    - `num_images > 0` → synthesized user gets `<image>\\n` × num_images,
      and any `<image>` tokens in non-user messages are stripped to
      avoid the forward pass's placeholder double-count (images already
      land in the synthesized user via the processor).
    - Existing user messages are never touched here; leave image-count
      repair to `align_image_placeholders`.

    Returns a new list; inputs are never mutated.
    """
    has_user = any(m.get("role") == "user" for m in messages)
    if has_user:
        return messages

    repaired = [dict(m) for m in messages]
    placeholder_content = ("<image>\n" * num_images) if num_images > 0 else ""
    stripped = 0
    if num_images > 0:
        for m in repaired:
            content = m.get("content")
            if isinstance(content, str) and "<image>" in content:
                stripped += content.count("<image>")
                m["content"] = content.replace("<image>", "")
    repaired.insert(0, {"role": "user", "content": placeholder_content})
    tag = f"[ensure-user{' ' + sample_id if sample_id else ''}]"
    extras = []
    if num_images > 0:
        extras.append(f"{num_images} <image> placeholder(s)")
    if stripped > 0:
        extras.append(f"stripped {stripped} <image> marker(s) from non-user turns")
    detail = "; ".join(extras) if extras else "no <image> placeholders"
    logger.info("%s no user message; synthesized one (%s).", tag, detail)
    return repaired


def align_image_placeholders(
    messages: list[dict],
    num_images: int,
    *,
    sample_id: str | None = None,
) -> list[dict]:
    """Ensure user-message `<image>` count equals `num_images`.

    Training data occasionally ships `images=[...]` without matching
    `<image>` placeholders in user content (or the reverse). Without repair
    the Qwen processor silently emits pixel_values for every image while
    the prompt carries no `<|image_pad|>`, and forward dies with:

        ValueError: Image features and image tokens do not match,
                    tokens: 0, features: 256

    Repair policy (LLaVA/InternVL convention):
    - `placeholders < num_images`: prepend the missing `<image>\\n` chunks
      to the first user message with string content.
    - `placeholders > num_images`: strip the trailing excess from user
      messages, iterating in reverse.

    Returns a new messages list; inputs are never mutated. Non-string
    content (OpenAI multi-modal list form) is not counted or repaired.
    """
    count = _count_image_placeholders(messages)
    if count == num_images:
        return messages

    tag = f"[align{' ' + sample_id if sample_id else ''}]"
    repaired = [dict(m) for m in messages]

    if count < num_images:
        missing = num_images - count
        for i, m in enumerate(repaired):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                repaired[i]["content"] = ("<image>\n" * missing) + m["content"]
                logger.warning(
                    "%s prepended %d <image> placeholder(s) to first user message "
                    "(had %d, needed %d).",
                    tag, missing, count, num_images,
                )
                return repaired
        logger.warning(
            "%s %d image(s) but no string-content user message; leaving as-is "
            "(downstream may still error).",
            tag, num_images,
        )
        return repaired

    excess = count - num_images
    for i in range(len(repaired) - 1, -1, -1):
        if excess == 0:
            break
        if repaired[i].get("role") != "user":
            continue
        content = repaired[i].get("content")
        if not isinstance(content, str):
            continue
        while excess > 0:
            idx = content.rfind("<image>")
            if idx == -1:
                break
            content = content[:idx] + content[idx + len("<image>"):]
            excess -= 1
        repaired[i]["content"] = content
    stripped = count - num_images - excess
    logger.warning(
        "%s stripped %d excess <image> placeholder(s) from user messages "
        "(had %d, needed %d).",
        tag, stripped, count, num_images,
    )
    return repaired


def normalize_qwen_message_content(
    content: Any,
    vision_start: str,
    image_token: str,
    vision_end: str,
) -> Any:
    """Expand `<image>` into Qwen vision tokens; pass non-string content through."""
    if not isinstance(content, str):
        return content
    return content.replace("<image>", f"{vision_start}{image_token}{vision_end}")


def last_assistant_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    raise ValueError("Sample has no assistant message; cannot build prompt/completion.")


def normalize_messages(
    messages: list[dict],
    *,
    vision_start: str,
    image_token: str,
    vision_end: str,
) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        m2 = dict(m)
        m2["content"] = normalize_qwen_message_content(
            m.get("content"), vision_start, image_token, vision_end,
        )
        out.append(m2)
    return out


def render_prompt_and_completion(
    processing_class,
    messages: list[dict],
    *,
    vision_start: str,
    image_token: str,
    vision_end: str,
) -> tuple[str, str]:
    """Render prompt + completion with two `apply_chat_template` calls.

    Prompt = messages before the last assistant turn, with generation prompt.
    Completion = the last assistant turn only, obtained by slicing the full
    render. Falls back to raw content + EOS if the template produces identical
    prompt/full strings (e.g. templates that don't emit a generation prompt).
    """
    last_idx = last_assistant_index(messages)
    full_msgs = normalize_messages(
        messages[: last_idx + 1],
        vision_start=vision_start,
        image_token=image_token,
        vision_end=vision_end,
    )
    prompt_msgs = full_msgs[:-1]
    prompt_text = processing_class.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    full_text = processing_class.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False,
    )
    completion_text = full_text[len(prompt_text):]
    if not completion_text:
        completion_text = normalize_qwen_message_content(
            messages[last_idx].get("content"),
            vision_start,
            image_token,
            vision_end,
        )
        eos = processing_class.tokenizer.eos_token
        if eos and not completion_text.endswith(eos):
            completion_text += eos
    return prompt_text, completion_text


def load_sample_images(paths: list[str]) -> list[PIL.Image.Image]:
    """Open + convert + force-decode images so that pixels live in this
    process's memory (pass-through to the main training loop doesn't trigger
    a second decode)."""
    images: list[PIL.Image.Image] = []
    for p in paths:
        img = PIL.Image.open(p).convert("RGB")
        img.load()
        images.append(img)
    return images
