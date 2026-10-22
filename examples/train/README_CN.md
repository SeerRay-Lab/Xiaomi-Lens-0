# 训练数据格式

[English](README.md) | 简体中文

`train.py` 读取 YAML 清单中列出的 JSONL 文件，每行对应一个样本。以下命令均在仓库根目录执行。

## 数据清单

```yaml
# 仅用于训练的数据清单
datasets:
  - name: refcoco            # 可选：用于展示的数据集名称
    path: data/refcoco.jsonl # 相对路径以当前清单文件所在目录为基准
    image_folder: images     # 可选：相对图像路径的前缀
    tag: segmentation        # 可选：在训练预检报告中展示的自定义标签
    sampling_strategy: all   # all | first:N | end:N | random:N | random:P%
```

如需在训练过程中评估，将数据条目分别放入 `train:` 和 `eval:` 下，参考 [`stage2.yaml`](stage2.yaml)，并传入 `--eval_strategy steps --eval_steps N`。

## 样本字段

| 字段 | 说明 |
|---|---|
| `messages` | 对话轮次，每轮包含 `role`（`system`、`user` 或 `assistant`）和字符串 `content`。每个 assistant 轮次都参与 LM loss 监督。 |
| `images` | 可选的图像路径列表。user 轮次中的每个 `<image>` 占位符依次对应下一张图像。缺失或多余的占位符会自动修正。 |
| `segmentation` | 可选的 COCO RLE 对象列表，格式为 `{"size": [height, width], "counts": ...}`。第 *k* 项监督所有 assistant 轮次中按阅读顺序出现的第 *k* 个 `<|mask|>`。`counts` 可以是压缩字符串或整数列表。 |

格式要求：

- `<|mask|>` 仅出现在 assistant 轮次中。每个 `<|mask|>` 对应其前方最近的一张图像，RLE 的 `size` 必须等于该图像的原始 `[height, width]`。
- 边界框坐标为 `[0, 1000]` 范围内的千分比整数。一个定位实例写作 `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`。
- assistant 内容以 `<think>\n\n</think>\n\n`（空推理）或 `<think>\n{reasoning}\n</think>\n\n` 开头。

## 样本结构

Stage 1（掩码解码器预热）使用仅包含 assistant 轮次的样本，程序会自动补充承载图像的 user 轮次：

```json
{
  "images": ["0000.png"],
  "messages": [
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe mask of all regions matching the description \"red circle\" in <image> is <|mask|>."}
  ],
  "segmentation": [{"size": [240, 320], "counts": "..."}]
}
```

Stage 2（监督微调）混合定位、视觉问答（VQA）和纯文本样本：

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

不含 `segmentation` 的样本仅使用 LM loss 训练。多轮对话可在每轮包含一个 `<|mask|>`，`segmentation` 按阅读顺序列出对应掩码。

## 示例数据

[`sample/`](sample) 包含两条可直接训练的 RefCOCO 指代表达分割样本及其 COCO 图像。除图像路径外，样本与训练集中的原始内容一致。每条样本包含一个压缩 RLE 掩码，[`sample/sample.yaml`](sample/sample.yaml) 可用于任一阶段的启动脚本：

```bash
NPROC_PER_NODE=1 MODEL_PATH=checkpoints/Qwen3.5-4B DATA_CONFIG=examples/train/sample/sample.yaml \
  bash scripts/train_stage1.sh --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_steps 2
```

| 图像 | 指代表达 | 来源 | 许可 |
|---|---|---|---|
| `COCO_train2014_000000581739.jpg` | "RIGHT VASE" | [COCO](http://images.cocodataset.org/train2014/COCO_train2014_000000581739.jpg), [Flickr](http://farm3.staticflickr.com/2582/3858684718_c8d919054d_z.jpg) | [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/) |
| `COCO_train2014_000000580562.jpg` | "gray dump truck" | [COCO](http://images.cocodataset.org/train2014/COCO_train2014_000000580562.jpg), [Flickr](http://farm9.staticflickr.com/8193/8365824259_8a0e22a6ef_z.jpg) | [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/) |

图像来自 [COCO](https://cocodataset.org/)，指代表达来自 [RefCOCO](https://github.com/lichengunc/refer)，均保留原始许可。

## 合成玩具数据

[`make_toy_data.py`](make_toy_data.py) 生成一个由彩色形状组成的小型合成数据集，覆盖上述各类样本结构，可用于端到端检查安装环境：

```bash
python examples/train/make_toy_data.py --output examples/train/toy --num-images 48
```

输出为 `examples/train/toy/{images,stage1.jsonl,stage2.jsonl}`，[`stage1.yaml`](stage1.yaml) 和 [`stage2.yaml`](stage2.yaml) 引用了这些文件。
