# Repository Guidelines

## Project Overview

Xiaomi-Lens-0 extends Qwen3.5 with mask-token segmentation. This repository contains the inference package, usage examples, documentation, and project homepage.

## Documentation & Presentation

Write clear, concise, public-facing documentation using repository-relative paths and reproducible commands. Keep AGENTS.md focused on actionable project guidance, with direct, affirmative language and comment-free Markdown.

Use these projects as presentation references:

- [Xiaomi-OCR-0](https://github.com/SeerRay-Lab/Xiaomi-OCR-0): model introduction, capabilities, setup, demos, performance tables, and citation.
- [DareBench](https://github.com/SeerRay-Lab/DareBench): title, tagline, navigation, release news, overview visuals, benchmark results, and project links.

Make the project identity, core capabilities, visual examples, report, model, demo, evaluation results, quick start, and citation easy to find. Use verified project content and working links. Keep the GitHub README and standalone homepage aligned.

## Project Structure

- `README.md` and `README_CN.md`: project introduction and usage documentation.
- `docs/`: project homepage and visual assets.
- `src/xiaomi_lens/model/`: Qwen3.5 model integration and `PixelMaskDecoder`. `modeling_xiaomi_lens.py` defines `XiaomiLensQwen3_5ForConditionalGeneration`.
- `src/xiaomi_lens/integrations/vllm/`: text and pooling-mask serving, registered through `pyproject.toml` entry points.
- `infer.py`: Hugging Face inference CLI and two-phase decoding shared with serving.
- `tools/`: scripts for rendering homepage assets.

Use `XiaomiLensQwen3_5ForConditionalGeneration` as the checkpoint architecture and vLLM registration name. Keep `src/` focused on inference.

## Development Commands

- `pip install -e .`: install the inference package.
- `pip install -e ".[vllm]"`: install with serving dependencies.
- `python infer.py --help`: inspect inference arguments.

## Coding & Validation

Match the existing Python style: four-space indentation, `snake_case` functions and modules, and descriptive class names. Keep changes focused and follow directory-specific project guidance.

Validate changed behavior with relevant checks. Use pytest and `test_*.py` naming for Python tests. Inspect rendered pages and assets after visual changes.

## Commits & Pull Requests

Keep commits focused and use imperative subjects. Pull requests should describe the change, link relevant issues, and summarize validation. Include screenshots or page references for visual changes.
