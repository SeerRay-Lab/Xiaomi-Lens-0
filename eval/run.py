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

"""LensBench evaluation with HF, parallel GPUs, offline vLLM and mask comparison."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infer import (
    build_inputs, generate_completion, load_model_and_processor, run_forward,
    split_answer_content,
)
from xiaomi_lens.constants import DEFAULT_MAX_NEW_TOKENS, MASK_TOKEN

# Preserve the dataset instruction and append the requested output format.
PROMPT = "{instruction}\nReport bbox coordinates and masks."
GROUPS = ("domain", "explicitness", "cardinality")
EXPECTED_ROWS = 4800
ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 1800
DEFAULT_MODEL = "SeerRay-Lab/Xiaomi-Lens-0"
DEFAULT_DATASET = "SeerRay-Lab/LensBench"
DATASET_FILE = "test.parquet"


def resolve_model(value):
    """Return a local checkpoint directory, downloading a Hub model ID if needed."""
    if Path(value).is_dir():
        return Path(value).resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(value))


def resolve_dataset(value):
    """Return the local LensBench parquet file for a file, directory or Hub dataset ID."""
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        return (path / DATASET_FILE).resolve()
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(value, DATASET_FILE, repo_type="dataset"))


def predict_hf(model, processor, mask_token_id, image, text, device, max_new_tokens):
    """Generate one completion and return answer-mask logits cropped to the image canvas."""
    inputs, prompt = build_inputs(processor, image, text, device)
    ids, generated, total = generate_completion(
        model, processor, inputs, max_new_tokens, mask_token_id)
    start, count = answer_mask_range(generated, total, prompt)
    logits = None
    if count:
        logits = run_forward(model, inputs, ids, mask_token_id, device)
        if logits.shape[1] != total:
            raise ValueError("Forward mask count differs from completion")
        grid_h, grid_w = inputs["image_grid_thw"][0, 1:].tolist()
        logits = logits[:, start:start + count, :grid_h * 16, :grid_w * 16]
    return generated, count, logits


def score_masks(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(f"Mask shape mismatch: {prediction.shape} vs {target.shape}")
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return {"intersection": intersection, "union": union,
            "iou": intersection / union if union else 1.0}


def answer_mask_range(text, total_masks, prompt_text=""):
    # A chat template can open thinking before generation starts. If generation
    # stops before </think>, every generated mask still belongs to reasoning.
    assistant_prefix = prompt_text.rsplit("<|im_start|>assistant", 1)[-1]
    if "</think>" not in text and (
        "<think>" in text or (
            "<think>" in assistant_prefix and "</think>" not in assistant_prefix
        )
    ):
        return total_masks, 0
    answer, thought_count = split_answer_content(text)
    answer_count = answer.count(MASK_TOKEN)
    if thought_count + answer_count != total_masks:
        raise ValueError("Decoded mask count differs from generated token count")
    return thought_count, answer_count


def merge_masks(logits, size, start=0, count=None):
    """Resize raw logits to (height, width), threshold, then union instances."""
    masks = logits[0, start:] if count is None else logits[0, start:start + count]
    if masks.shape[0] == 0:
        return np.zeros(size, dtype=bool)
    masks = F.interpolate(masks[:, None].float(), size=size, mode="bilinear",
                          align_corners=False)[:, 0]
    return (masks.sigmoid() >= 0.5).any(dim=0).cpu().numpy()


def decode_row(row):
    from pycocotools import mask as coco_mask

    data = row["image"]["bytes"]
    if not data:
        raise ValueError("LensBench requires embedded image bytes")
    image = Image.open(io.BytesIO(data)).convert("RGB")
    rles = row["segmentation"]
    if isinstance(rles, str):
        rles = json.loads(rles)
    if isinstance(rles, dict):
        rles = [rles]
    target = np.zeros((image.height, image.width), dtype=bool)
    for rle in rles:
        rle = dict(rle)
        if isinstance(rle["counts"], list):
            rle = coco_mask.frPyObjects(rle, *rle["size"])
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        decoded = coco_mask.decode(rle).astype(bool)
        if decoded.shape != target.shape:
            raise ValueError(f"RLE/image size mismatch: {decoded.shape}, {target.shape}")
        target |= decoded
    return image, target


def summarize(records):
    def metrics(items):
        n = len(items)
        union = sum(r["union"] for r in items)
        return {
            "count": n,
            "Prec@0.5": sum(r["iou"] >= 0.5 for r in items) / n if n else None,
            "cIoU": sum(r["intersection"] for r in items) / union if union else (1.0 if n else None),
            "gIoU": sum(r["iou"] for r in items) / n if n else None,
        }
    result = {"overall": metrics(records)}
    for group in GROUPS:
        labels = sorted({r[group] for r in records})
        result[group] = {label: metrics([r for r in records if r[group] == label])
                         for label in labels}
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare_output(output, config, resume):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    predictions = output / "predictions.jsonl"
    if not resume:
        if any((output / name).exists() for name in
               ("config.json", "predictions.jsonl", "metrics.json", "error.json")):
            raise ValueError("Output already contains a run; use --resume or a new directory")
        write_json(config_path, config)
        return {}
    saved_raw = json.loads(config_path.read_text())
    # Runs created before backend selection used HF exclusively.
    saved_config = {"backend": "hf", **saved_raw}
    # Earlier runs recorded batch_size; single-sample decoding matches batch size 1.
    if saved_config.pop("batch_size", 1) != 1:
        raise ValueError("Resume requires a run created with batch size 1")
    checked_config = {"backend": "hf", **config}
    if saved_config != checked_config:
        raise ValueError("Resume configuration mismatch (model, dataset or inference parameters)")
    if saved_raw != config:
        write_json(config_path, config)
    records = {}
    if predictions.exists():
        # Only an unterminated final line may be discarded after interruption.
        with predictions.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                record = json.loads(line)
                index = record["index"]
                if index in records:
                    raise ValueError(f"Duplicate completed index: {index}")
                i, u, iou = record["intersection"], record["union"], record["iou"]
                if not (0 <= i <= u and math.isfinite(iou) and
                        math.isclose(iou, i / u if u else 1.0)):
                    raise ValueError(f"Invalid saved metrics for index {index}")
                records[index] = record
    return records


def print_metrics(result):
    print(f"\n{'Group / label':36} {'N':>6} {'Prec@0.5':>10} {'cIoU':>10} {'gIoU':>10}")
    rows = [("overall", result["overall"])]
    rows += [(f"{g} / {label}", m) for g in GROUPS for label, m in result[g].items()]
    for label, m in rows:
        values = [f"{100 * m[k]:.2f}" if m[k] is not None else "n/a"
                  for k in ("Prec@0.5", "cIoU", "gIoU")]
        print(f"{label:36} {m['count']:6d} " + " ".join(f"{v:>10}" for v in values))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET,
                        help="LensBench parquet file, its directory, or Hugging Face dataset ID")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lensbench"))
    parser.add_argument("--device", default="cuda:0", help="HF device")
    parser.add_argument("--devices", nargs="+", help="Run one HF shard per device, then merge")
    parser.add_argument("--compare-postprocessing", action="store_true",
                        help="Compare bilinear and nearest mask postprocessing on a sample")
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-text-device", default="cuda:0")
    parser.add_argument("--vllm-mask-device", default="cuda:1")
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--max-new-tokens and --limit must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num-shards > 0 and 0 <= shard-index < num-shards")
    if args.devices:
        if (len(set(args.devices)) != len(args.devices) or args.backend != "hf" or
                args.num_shards != 1 or args.shard_index != 0 or args.compare_postprocessing):
            parser.error("--devices requires distinct devices, HF, no manual sharding or comparison")
    if args.compare_postprocessing:
        if args.backend != "hf" or args.resume or args.limit is not None:
            parser.error("--compare-postprocessing requires HF, a fresh run and no --limit")
        if args.samples_per_group <= 0:
            parser.error("--samples-per-group must be positive")
    if args.backend == "vllm":
        try:
            validate_options(args)
        except ValueError as error:
            parser.error(str(error))
    try:
        args.model_path = resolve_model(args.model_path)
        args.dataset_path = resolve_dataset(args.dataset_path)
    except Exception as error:
        parser.error(f"Cannot resolve model or dataset: {error}")
    return args


def save_record(output, record, records, count):
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()
    os.fsync(output.fileno())
    records[record["index"]] = record
    print(f"[{len(records)}/{count}] index={record['index']} "
          f"masks={record['pred_mask_count']} IoU={record['iou']:.4f}", flush=True)


def main():
    import pyarrow.parquet as pq

    args = parse_args()
    if args.devices:
        return run_parallel(args)
    if args.compare_postprocessing:
        return run_comparison(args)
    parquet = pq.ParquetFile(args.dataset_path)
    total = parquet.metadata.num_rows
    if total != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} LensBench rows, found {total}")
    target_count = min(total, args.limit) if args.limit else total
    shard_count = len(range(args.shard_index, target_count, args.num_shards))
    device = torch.device(args.device)
    attention = ("flash_attention_2" if device.type == "cuda" and
                 importlib.util.find_spec("flash_attn") else "sdpa")
    config = {
        "backend": args.backend,
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_size": args.dataset_path.stat().st_size,
        "dataset_mtime_ns": args.dataset_path.stat().st_mtime_ns,
        "dataset_rows": total, "device": str(device),
        "max_new_tokens": args.max_new_tokens, "limit": args.limit,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "attention": attention, "prompt": PROMPT, "do_sample": False,
        "threshold": 0.5, "empty_iou": 1.0, "format_version": 1,
    }
    if args.backend == "vllm":
        config.update(device=None, attention="vllm-auto", vllm=backend_config(args))
    records = prepare_output(args.output_dir, config, args.resume)
    vllm_backend = None
    context = {"stage": "model_loading"}
    try:
        model = processor = mask_token_id = None
        seen = set()
        dataset_seen = set()
        def pending_rows():
            position = 0
            for batch in parquet.iter_batches(batch_size=1):
                if position >= target_count:
                    break
                row = batch.to_pylist()[0]
                index = row["index"]
                if index in dataset_seen:
                    raise ValueError(f"Duplicate dataset index {index}")
                dataset_seen.add(index)
                position += 1
                if (position - 1) % args.num_shards != args.shard_index:
                    continue
                seen.add(index)
                if index in records:
                    if any(records[index][g] != row[g] for g in GROUPS):
                        raise ValueError(f"Saved group mismatch for index {index}")
                    continue
                yield row

        with (args.output_dir / "predictions.jsonl").open("a") as output:
            for row in pending_rows():
                context = {"stage": "sample",
                           **{k: row[k] for k in ("index", "instruction", *GROUPS)}}
                text = PROMPT.format(instruction=row["instruction"])
                image, target = decode_row(row)
                if args.backend == "vllm":
                    if vllm_backend is None:
                        vllm_backend = VLLMBackend(args.model_path, args.output_dir, config["vllm"])
                    result = vllm_backend.predict(row["image"]["bytes"], text, args.max_new_tokens)
                    prediction = result.pop("prediction")
                else:
                    if model is None:
                        model, processor, mask_token_id = load_model_and_processor(
                            args.model_path, device, attn_implementation=attention)
                    generated, count, logits = predict_hf(
                        model, processor, mask_token_id, image, text, device, args.max_new_tokens)
                    prediction = (merge_masks(logits, target.shape) if count
                                  else np.zeros(target.shape, dtype=bool))
                    result = {"generated_text": generated, "pred_mask_count": count}
                    del logits
                record = {"index": row["index"], **{g: row[g] for g in GROUPS},
                          **result, **score_masks(prediction, target)}
                save_record(output, record, records, shard_count)
        if set(records) != seen or len(seen) != shard_count:
            raise ValueError("Completed IDs do not exactly cover the requested dataset rows")
        result = summarize(list(records.values()))
        result.update({"config": config, "complete": args.num_shards == 1 and len(seen) == EXPECTED_ROWS,
                       "shard_complete": True, "shard_expected_count": shard_count,
                       "expected_count": EXPECTED_ROWS, "evaluated_count": len(seen)})
        write_json(args.output_dir / "metrics.json", result)
        (args.output_dir / "error.json").unlink(missing_ok=True)
        print_metrics(result)
        print(f"Full LensBench complete: {result['complete']}", flush=True)
    except Exception:
        write_json(args.output_dir / "error.json", {
            **context,
            "backend_context": vllm_backend.context if vllm_backend is not None else None,
            "traceback": traceback.format_exc(), "config": config,
        })
        raise
    finally:
        if vllm_backend is not None:
            vllm_backend.close()


def checkpoint_identity(checkpoint):
    names = ('config.json', 'generation_config.json', 'processor_config.json',
             'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
             'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
             'model.safetensors.index.json')
    return {
        'metadata_sha256': {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                            for name in names if (checkpoint / name).is_file()},
        'weight_sizes': {p.name: p.stat().st_size for p in sorted(checkpoint.glob('*.safetensors'))},
    }


def validate_options(args):
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError('vLLM requires --num-shards 1 --shard-index 0')
    devices = (args.vllm_text_device, args.vllm_mask_device)
    if any(re.fullmatch(r'cuda:\d+', d) is None for d in devices):
        raise ValueError('vLLM devices must use cuda:N notation')
    if int(devices[0].split(':')[1]) == int(devices[1].split(':')[1]):
        raise ValueError('Text and mask engines require different CUDA devices')
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        raise ValueError('vLLM GPU memory utilization must be between 0 and 1')
    if args.vllm_max_model_len <= args.max_new_tokens:
        raise ValueError('vLLM context length must leave room for the image and prompt')


def backend_config(args):
    validate_options(args)
    return {
        'text_device': args.vllm_text_device, 'mask_device': args.vllm_mask_device,
        'max_model_len': args.vllm_max_model_len,
        'gpu_memory_utilization': args.vllm_gpu_memory_utilization,
        'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'max_num_seqs': 1,
        'seed': 0, 'temperature': 0.0, 'top_p': 1.0, 'top_k': -1,
        'mask_cudagraph_capture_sizes': [1],
        'version': importlib.metadata.version('vllm'),
        'checkpoint_identity': checkpoint_identity(args.model_path.resolve()),
        'implementation_version': 1,
    }


def _visible_gpu(device):
    index = int(device.split(':')[1])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(index)
    devices = visible.split(',')
    if index >= len(devices) or not devices[index].strip():
        raise ValueError(f'{device} is outside CUDA_VISIBLE_DEVICES={visible}')
    return devices[index].strip()


def _prepare_views(checkpoint, cache):
    from xiaomi_lens.integrations.vllm.prepare_served import prepare

    for name, tokenizer_only in (('text', False), ('mask', True)):
        directory = cache / name
        if not (directory / 'tokenizer_config.json').is_file():
            prepare(checkpoint, directory, tokenizer_only=tokenizer_only)
    # Same entry points as the serving launcher, scoped to these child processes.
    metadata = cache / 'plugins' / 'xiaomi_lens-0.1.0.dist-info'
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: xiaomi-lens\nVersion: 0.1.0\n')
    (metadata / 'entry_points.txt').write_text(
        '[vllm.general_plugins]\n'
        'xiaomi_lens_register = xiaomi_lens.integrations.vllm.plugin:register\n\n'
        '[vllm.io_processor_plugins]\n'
        'xiaomi_lens_mask = xiaomi_lens.integrations.vllm.io_processor:make_io_processor\n')


def _engine_options(cache, phase, config):
    options = dict(model=str(cache / phase), dtype=config['dtype'],
                   tensor_parallel_size=1, max_num_seqs=1,
                   max_model_len=config['max_model_len'],
                   gpu_memory_utilization=config['gpu_memory_utilization'],
                   seed=0, trust_remote_code=True)
    if phase == 'text':
        options.update(runner='generate', hf_overrides={
            'architectures': ['Qwen3_5ForConditionalGeneration']})
    else:
        options.update(runner='pooling', io_processor_plugin='xiaomi_lens_mask',
                       enable_chunked_prefill=False, enable_prefix_caching=False,
                       mm_processor_cache_gb=0,
                       compilation_config={'cudagraph_capture_sizes': [1],
                                           'max_cudagraph_capture_size': 1})
    return options


def _pool_masks(engine, request):
    from vllm import PoolingParams

    return engine.encode({'data': request}, PoolingParams(task='plugin'),
                         pooling_task='plugin', use_tqdm=False)[0].outputs


def _worker(connection, phase, gpu, cache, config, log_path):
    os.setsid()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # Absolute-python invocations need the selected venv's JIT tools (e.g. ninja).
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    paths = [str(cache / 'plugins'), str(ROOT / 'src'), str(ROOT)]
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + [os.environ.get('PYTHONPATH', '')])
    sys.path[:0] = paths
    with open(log_path, 'a', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        engine = None
        try:
            from vllm import LLM, SamplingParams
            from xiaomi_lens.constants import MASK_TOKEN

            options = _engine_options(cache, phase, config)
            engine = LLM(**options)
            tokenizer = engine.get_tokenizer()
            mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
            connection.send(('ok', {'mask_token_id': mask_id}))
            while True:
                request = connection.recv()
                if request is None:
                    break
                if phase == 'text':
                    image = Image.open(io.BytesIO(request['image_bytes'])).convert('RGB')
                    messages = [{'role': 'user', 'content': [
                        {'type': 'image'}, {'type': 'text', 'text': request['text']}]}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                            max_tokens=request['max_new_tokens'],
                                            skip_special_tokens=False, seed=0)
                    output = engine.generate({'prompt': prompt, 'multi_modal_data': {'image': image}},
                                             params, use_tqdm=False)[0]
                    if len(output.prompt_token_ids) + request['max_new_tokens'] > config['max_model_len']:
                        raise ValueError('Prompt plus generation limit exceeds vLLM context length')
                    completion = output.outputs[0]
                    response = {'generated_text': completion.text,
                                'completion_token_ids': list(completion.token_ids),
                                'prompt_text': prompt}
                else:
                    response = _pool_masks(engine, request)
                connection.send(('ok', response))
        except BaseException:
            try:
                connection.send(('error', traceback.format_exc()))
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            if engine is not None:
                core = getattr(engine.llm_engine, 'engine_core', None)
                if core is not None and hasattr(core, 'shutdown'):
                    core.shutdown()
            connection.close()


def decode_pooling_masks(response, expected_count, size):
    from pycocotools import mask as coco_mask

    if (not isinstance(response, dict) or response.get('num_masks') != expected_count or
            response.get('mask_space') != 'original' or
            response.get('mask_size') != [size[1], size[0]] or
            len(response.get('masks_rle', [])) != expected_count):
        raise ValueError('Pooling response mask count, space or dimensions mismatch')
    prediction = np.zeros(size, dtype=bool)
    for rle in response['masks_rle']:
        rle = dict(rle)
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
        mask = coco_mask.decode(rle).astype(bool)
        if mask.shape != size:
            raise ValueError('Decoded pooling RLE dimensions mismatch')
        prediction |= mask
    return prediction


class VLLMBackend:
    def __init__(self, checkpoint, output, config):
        self.workers = {}
        self.context = {'stage': 'initialization'}
        try:
            cache = output.resolve() / '.vllm'
            _prepare_views(checkpoint.resolve(), cache)
            ctx = mp.get_context('spawn')
            mask_ids = []
            for phase in ('text', 'mask'):
                gpu = _visible_gpu(config[f'{phase}_device'])
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(
                    child, phase, gpu, cache, config, str(output.resolve() / f'vllm-{phase}.log')))
                process.start()
                child.close()
                self.workers[phase] = (process, parent)
            for phase in ('text', 'mask'):
                mask_ids.append(self._receive(phase)['mask_token_id'])
            if mask_ids[0] != mask_ids[1]:
                raise ValueError('Text and pooling tokenizers disagree on mask token ID')
            self.mask_token_id = mask_ids[0]
        except BaseException:
            self.close()
            raise

    def _receive(self, phase):
        process, connection = self.workers[phase]
        deadline = time.monotonic() + TIMEOUT_SECONDS
        while not connection.poll(.2):
            if not process.is_alive():
                raise RuntimeError(f'vLLM {phase} worker exited with code {process.exitcode}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'vLLM {phase} worker exceeded {TIMEOUT_SECONDS}s')
        try:
            status, value = connection.recv()
        except EOFError as error:
            raise RuntimeError(f'vLLM {phase} worker disconnected') from error
        if status != 'ok':
            raise RuntimeError(f'vLLM {phase} worker failed:\n{value}')
        return value

    def _request(self, phase, payload):
        self.context['stage'] = phase
        self.workers[phase][1].send(payload)
        return self._receive(phase)

    def predict(self, image_bytes, text, max_new_tokens):
        self.context = {'stage': 'text'}
        generated = self._request('text', dict(image_bytes=image_bytes, text=text,
                                             max_new_tokens=max_new_tokens))
        self.context.update(generated)
        ids = generated['completion_token_ids']
        _, count = answer_mask_range(generated['generated_text'], ids.count(self.mask_token_id),
                                     generated['prompt_text'])
        with Image.open(io.BytesIO(image_bytes)) as image:
            size = (image.height, image.width)
        prediction = np.zeros(size, dtype=bool)
        if count:
            response = self._request('mask', {
                'image_b64': base64.b64encode(image_bytes).decode('ascii'), 'text': text,
                'completion': generated['generated_text'], 'completion_token_ids': ids,
                'threshold': .5, 'mask_space': 'original', 'output_format': 'rle',
                'return_overlay': False,
            })
            prediction = decode_pooling_masks(response, count, size)
        return {'prediction': prediction, 'generated_text': generated['generated_text'],
                'completion_token_ids': ids, 'pred_mask_count': count}

    def close(self):
        for process, connection in self.workers.values():
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
                process.join(timeout=10)
            # Each worker owns a new session, including its EngineCore descendants.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if process.is_alive():
                    process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
                process.join(timeout=5)
            connection.close()
        self.workers.clear()


def merge_results(output, shard_count):
    import pyarrow.parquet as pq

    records = {}
    configs = []
    for shard in range(shard_count):
        directory = output / f"shard-{shard}"
        config = json.loads((directory / "config.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        if (config["num_shards"] != shard_count or config["shard_index"] != shard or
                not metrics["shard_complete"] or metrics["config"] != config):
            raise ValueError(f"Incomplete or inconsistent shard {shard}")
        comparable = {k: v for k, v in config.items() if k not in ("device", "shard_index")}
        if configs and comparable != configs[0]:
            raise ValueError("Shard configurations differ")
        configs.append(comparable)
        shard_records = []
        for line in (directory / "predictions.jsonl").read_text().splitlines():
            record = json.loads(line)
            index = record["index"]
            if index in records:
                raise ValueError(f"Duplicate prediction index {index}")
            records[index] = (shard, record)
            shard_records.append(record)
        if summarize(shard_records) != {k: metrics[k] for k in ("overall", *GROUPS)}:
            raise ValueError(f"Saved shard metrics disagree with predictions: {shard}")
    config = configs[0]
    dataset = Path(config["dataset_path"])
    if (dataset.stat().st_size != config["dataset_size"] or
            dataset.stat().st_mtime_ns != config["dataset_mtime_ns"]):
        raise ValueError("Dataset changed during evaluation")
    parquet = pq.ParquetFile(dataset)
    if parquet.metadata.num_rows != EXPECTED_ROWS:
        raise ValueError("Dataset must contain all 4,800 rows")
    count = min(EXPECTED_ROWS, config["limit"]) if config["limit"] else EXPECTED_ROWS
    expected = parquet.read(columns=["index", *GROUPS]).slice(0, count).to_pylist()
    ordered = []
    for position, row in enumerate(expected):
        index = row["index"]
        if index not in records:
            raise ValueError(f"Missing prediction index {index}")
        shard, record = records[index]
        if shard != position % shard_count or any(record[g] != row[g] for g in GROUPS):
            raise ValueError(f"Wrong shard or group for index {index}")
        ordered.append(record)
    if len(records) != count or len({r["index"] for r in expected}) != count:
        raise ValueError("Predictions must exactly cover the selected dataset IDs")
    temporary = output / "predictions.jsonl.tmp"
    with temporary.open("w") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output / "predictions.jsonl")
    config["devices"] = [json.loads((output / f"shard-{s}/config.json").read_text())["device"]
                         for s in range(shard_count)]
    result = summarize(ordered)
    result.update(config=config, complete=count == EXPECTED_ROWS,
                  expected_count=EXPECTED_ROWS, evaluated_count=count)
    write_json(output / "config.json", config)
    write_json(output / "metrics.json", result)
    print_metrics(result)
    print(f"Full LensBench complete: {result['complete']}", flush=True)
    return result


def run_parallel(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; use --resume to continue")
    processes = []
    logs = []
    try:
        for shard, device in enumerate(args.devices):
            command = [sys.executable, "-u", str(Path(__file__).with_name("run.py")),
                       "--model-path", str(args.model_path),
                       "--dataset-path", str(args.dataset_path),
                       "--output-dir", str(args.output_dir / f"shard-{shard}"),
                       "--device", device,
                       "--max-new-tokens", str(args.max_new_tokens),
                       "--num-shards", str(len(args.devices)), "--shard-index", str(shard)]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            if args.resume:
                command.append("--resume")
            log = (args.output_dir / f"shard-{shard}.log").open("a")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            print(f"shard={shard} device={device} pid={processes[-1].pid}", flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("An evaluation shard failed; inspect shard logs and error.json")
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("An evaluation shard failed; inspect shard logs and error.json")
        merge_results(args.output_dir, len(args.devices))
        (args.output_dir / "parallel-error.json").unlink(missing_ok=True)
    except BaseException:
        worker_statuses = [{"pid": p.pid, "returncode": p.poll()} for p in processes]
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        write_json(args.output_dir / "parallel-error.json", {
            "traceback": traceback.format_exc(), "workers": worker_statuses,
        })
        raise
    finally:
        for log in logs:
            log.close()


def nearest_binary_union(logits, size):
    """Match lensbench_eval: threshold logits >= 0, union, PIL nearest resize."""
    union = (logits[0].float() >= 0).any(dim=0).cpu().numpy()
    if union.shape != size:
        union = np.asarray(Image.fromarray(union.astype(np.uint8)).resize(
            (size[1], size[0]), Image.Resampling.NEAREST), dtype=bool)
    return union


def select_indices(rows, samples_per_group, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[g] for g in GROUPS)].append(row['index'])
    rng = random.Random(seed)
    selected = []
    for group in sorted(groups):
        if len(groups[group]) < samples_per_group:
            raise ValueError(f"Insufficient samples in group {group}")
        selected.extend(rng.sample(sorted(groups[group]), samples_per_group))
    return sorted(selected)


def comparison_summary(records):
    methods = {}
    for method in ('bilinear', 'nearest'):
        methods[method] = summarize([
            {**{g: r[g] for g in GROUPS}, **r[method]} for r in records
        ])
    delta = {}
    for section in ('overall', *GROUPS):
        def difference(a, b):
            return {k: a[k] - b[k] for k in ('Prec@0.5', 'cIoU', 'gIoU')}
        if section == 'overall':
            delta[section] = difference(methods['bilinear'][section], methods['nearest'][section])
        else:
            delta[section] = {label: difference(methods['bilinear'][section][label],
                                              methods['nearest'][section][label])
                              for label in methods['bilinear'][section]}
    changes = {
        'bilinear_higher_iou': sum(r['bilinear']['iou'] > r['nearest']['iou'] for r in records),
        'nearest_higher_iou': sum(r['nearest']['iou'] > r['bilinear']['iou'] for r in records),
        'equal_iou': sum(r['nearest']['iou'] == r['bilinear']['iou'] for r in records),
        'bilinear_only_pass_0.5': sum(r['bilinear']['iou'] >= .5 > r['nearest']['iou'] for r in records),
        'nearest_only_pass_0.5': sum(r['nearest']['iou'] >= .5 > r['bilinear']['iou'] for r in records),
        'different_masks': sum(r['disagreement_pixels'] > 0 for r in records),
        'max_absolute_iou_difference': max(abs(r['bilinear']['iou'] - r['nearest']['iou'])
                                           for r in records),
    }
    return {'count': len(records), 'methods': methods, 'bilinear_minus_nearest': delta,
            'paired_changes': changes, 'full_benchmark': False}


def merge_comparison(output, num_shards):
    records = []
    configs = []
    for shard in range(num_shards):
        directory = output / f'shard-{shard}'
        result = json.loads((directory / 'comparison.json').read_text())
        config = result['config']
        if config['num_shards'] != num_shards or config['shard_index'] != shard:
            raise ValueError('Invalid comparison shard identity')
        comparable = {k: v for k, v in config.items() if k not in ('shard_index', 'device')}
        if configs and comparable != configs[0]:
            raise ValueError('Comparison configurations differ')
        configs.append(comparable)
        rows = [json.loads(s) for s in (directory / 'predictions.jsonl').read_text().splitlines()]
        if [r['index'] for r in rows] != config['selected_indices'][shard::num_shards]:
            raise ValueError('Comparison shard does not cover its expected IDs')
        records.extend(rows)
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != configs[0]['selected_indices']:
        raise ValueError('Comparison IDs are missing or duplicated')
    result = {**comparison_summary(records), 'config': configs[0]}
    write_json(output / 'comparison.json', result)
    with (output / 'predictions.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return result


def run_comparison(args):
    import pyarrow.parquet as pq

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError('Comparison output directory must be empty')
    metadata = pq.read_table(args.dataset_path, columns=['index', *GROUPS]).to_pylist()
    selected = select_indices(metadata, args.samples_per_group, args.seed)
    wanted = set(selected[args.shard_index::args.num_shards])
    device = torch.device(args.device)
    attention = ('flash_attention_2' if device.type == 'cuda' and
                 importlib.util.find_spec('flash_attn') else 'sdpa')
    config = {'model_path': str(args.model_path.resolve()),
              'dataset_path': str(args.dataset_path.resolve()),
              'dataset_size': args.dataset_path.stat().st_size,
              'dataset_mtime_ns': args.dataset_path.stat().st_mtime_ns,
              'device': str(device), 'attention': attention,
              'samples_per_group': args.samples_per_group,
              'seed': args.seed, 'selected_indices': selected,
              'num_shards': args.num_shards, 'shard_index': args.shard_index,
              'prompt': PROMPT, 'max_new_tokens': args.max_new_tokens,
              'do_sample': False, 'threshold': .5}
    write_json(args.output_dir / 'config.json', config)
    context = {'stage': 'loading'}
    try:
        model, processor, mask_token = load_model_and_processor(args.model_path, device, attention)
        records = []
        with (args.output_dir / 'predictions.jsonl').open('w') as stream:
            for batch in pq.ParquetFile(args.dataset_path).iter_batches(batch_size=1):
                row = batch.to_pylist()[0]
                if row['index'] not in wanted:
                    continue
                context = {'index': row['index']}
                image, target = decode_row(row)
                text, count, logits = predict_hf(
                    model, processor, mask_token, image,
                    PROMPT.format(instruction=row['instruction']), device, args.max_new_tokens)
                bilinear = nearest = np.zeros(target.shape, dtype=bool)
                if count:
                    bilinear = merge_masks(logits, target.shape)
                    nearest = nearest_binary_union(logits, target.shape)
                del logits
                record = {'index': row['index'], **{g: row[g] for g in GROUPS},
                          'generated_text': text, 'pred_mask_count': count,
                          'bilinear': score_masks(bilinear, target),
                          'nearest': score_masks(nearest, target),
                          'disagreement_pixels': int(np.count_nonzero(bilinear != nearest)),
                          'image_pixels': int(target.size), 'gt_pixels': int(target.sum())}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                print(f"[{len(records)}/{len(wanted)}] index={row['index']} "
                      f"bilinear={record['bilinear']['iou']:.6f} nearest={record['nearest']['iou']:.6f}", flush=True)
        if {r['index'] for r in records} != wanted or len(records) != len(wanted):
            raise ValueError('Comparison sample coverage mismatch')
        write_json(args.output_dir / 'comparison.json', {**comparison_summary(records), 'config': config})
    except Exception:
        write_json(args.output_dir / 'error.json', {**context, 'traceback': traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
