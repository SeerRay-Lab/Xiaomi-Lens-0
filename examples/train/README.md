# Training data format

English | [简体中文](README_CN.md)

`train.py` reads JSONL files listed in a YAML manifest. Each line is one sample.

## Manifest

```yaml
# Train-only manifest
datasets:
  - name: refcoco            # optional display name
    path: data/refcoco.jsonl # relative paths resolve against this file's directory
    image_folder: images     # prefix for relative image paths; optional
    tag: segmentation        # optional free-form label shown in the preflight report
    sampling_strategy: all   # all | first:N | end:N | random:N | random:P%
```

To evaluate during training, put the entries under `train:` and `eval:` sections, as in [`stage2.yaml`](stage2.yaml), and pass `--eval_strategy steps --eval_steps N`.

## Sample fields

| Field | Description |
|---|---|
| `messages` | Chat turns with `role` (`system`, `user`, `assistant`) and string `content`. Every assistant turn is supervised by the LM loss. |
| `images` | Optional list of image paths. Each `<image>` placeholder in a user turn takes the next image. Missing or extra placeholders are repaired automatically. |
| `segmentation` | Optional list of COCO RLE objects `{"size": [height, width], "counts": ...}`. Entry *k* supervises the *k*-th `<|mask|>` across all assistant turns, in reading order. `counts` may be a compressed string or an integer list. |

Rules:

- `<|mask|>` appears only in assistant turns. Each `<|mask|>` binds to the nearest preceding image, and its RLE `size` must equal that image's original `[height, width]`.
- Boxes are per-mille integers in `[0, 1000]`. A grounded instance is written as `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`.
- Assistant content starts with `<think>\n\n</think>\n\n` (empty reasoning) or `<think>\n{reasoning}\n</think>\n\n`.

## Sample shapes

Stage 1 (mask decoder warmup) uses assistant-only rows. A user turn holding the images is added automatically:

```json
{
  "images": ["0000.png"],
  "messages": [
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe mask of all regions matching the description \"red circle\" in <image> is <|mask|>."}
  ],
  "segmentation": [{"size": [240, 320], "counts": "..."}]
}
```

Stage 2 (SFT) mixes grounding, VQA, and text rows:

```json
{
  "images": ["0000.png"],
  "messages": [
    {"role": "user", "content": "<image>\nLocate every shape in the image, report bbox coordinates and masks."},
    {"role": "assistant", "content": "<think>\nI see 2 shapes.\n</think>\n\n<bbox>[512, 321, 662, 521]</bbox><mask><|mask|></mask><bbox>[90, 120, 260, 400]</bbox><mask><|mask|></mask>"}
  ],
  "segmentation": [{"size": [240, 320], "counts": "..."}, {"size": [240, 320], "counts": "..."}]
}
```

Rows without `segmentation` train the LM loss only. Multi-turn rows may carry one `<|mask|>` per turn, and `segmentation` then lists them in reading order.

## Sample data

[`sample/`](sample) holds two ready-to-train RefCOCO referring-segmentation rows with their COCO images, exactly as they appear in the training set apart from the image paths. Each row has one compressed-RLE mask, so [`sample/sample.yaml`](sample/sample.yaml) works with either launcher:

```bash
NPROC_PER_NODE=1 MODEL_PATH=checkpoints/Qwen3.5-4B DATA_CONFIG=examples/train/sample/sample.yaml \
  bash scripts/train_stage1.sh --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_steps 2
```

| Image | Query | Source | License |
|---|---|---|---|
| `COCO_train2014_000000581739.jpg` | "RIGHT VASE" | [COCO](http://images.cocodataset.org/train2014/COCO_train2014_000000581739.jpg), [Flickr](http://farm3.staticflickr.com/2582/3858684718_c8d919054d_z.jpg) | [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/) |
| `COCO_train2014_000000580562.jpg` | "gray dump truck" | [COCO](http://images.cocodataset.org/train2014/COCO_train2014_000000580562.jpg), [Flickr](http://farm9.staticflickr.com/8193/8365824259_8a0e22a6ef_z.jpg) | [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/) |

Images are from [COCO](https://cocodataset.org/) and referring expressions from [RefCOCO](https://github.com/lichengunc/refer); both keep their original licenses.

## Toy data

[`make_toy_data.py`](make_toy_data.py) writes a small synthetic dataset of colored shapes that covers every shape above. Use it to check an installation end to end:

```bash
python examples/train/make_toy_data.py --output examples/train/toy --num-images 48
```

This writes `examples/train/toy/{images,stage1.jsonl,stage2.jsonl}`, which [`stage1.yaml`](stage1.yaml) and [`stage2.yaml`](stage2.yaml) reference.
