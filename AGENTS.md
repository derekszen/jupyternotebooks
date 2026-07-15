# Repository Guidelines

## Project Structure & Module Organization
Project metadata lives in `pyproject.toml`. Benchmark notebooks (`qwen_image_edit_benchmark.ipynb`, `qwen_image_ocr_experiment.ipynb`) and the batch runner `qwen_image_edit_benchmark.py` stay at the repository root for quick iteration. Shared helpers belong in `src/jupyternotebooks/` (create the package if absent) so notebooks can import rather than duplicate logic. Keep sample assets such as `data/input1.png` and `data/input2.png`, and place generated outputs in `results/` or `artifacts/`.

## Environment & Tooling
Use Python 3.10+ with `uv` for dependency management: `uv sync` installs runtime, diffusers, accelerate, and dev extras, while `uv run pre-commit install` enables lint hooks. Run on an Nvidia V100 (or similar CUDA GPU) and export `CUDA_VISIBLE_DEVICES=0` before starting kernels so diffusers binds to the expected device.

## Build, Test, and Development Commands
- `uv run python qwen_image_edit_benchmark.py --runs 5 --out results/benchmark.parquet`: execute the Qwen-Image-Edit benchmark and persist metrics.
- `uv run ruff check .` / `uv run ruff format .`: apply the lint and formatting profile (99-character lines, sorted imports).
- `uv run mypy .`: enforce strict typing on reusable modules.
- `uv run pytest`: execute the test suite once populated.

## Coding Style & Naming Conventions
Adopt four-space indentation, snake_case for modules, functions, and notebooks, and PascalCase for classes. Add type hints and brief docstrings to public helpers. Keep notebook cells focused on orchestration; move heavy lifting into importable modules. Reference assets with explicit relative paths such as `data/samples/ocr_input.png`.

## Benchmarking & Notebook Practices
Load `QwenImageEditPlusPipeline` with `torch_dtype=torch.bfloat16`, move it to `cuda`, and preload Pillow images before timing. Record each run with `time.perf_counter()` and aggregate timings into a Polars DataFrame before exporting to Parquet or CSV in `results/`. Clear execution counts and bulky outputs before commits, and note required environment variables or API keys in a small setup cell using safe defaults.

## Testing Guidelines
Organize tests by feature (e.g., `tests/ocr/test_pipeline.py`) and use pytest-style `test_*` functions. Start with deterministic utilities—prompt builders, metric calculators, file I/O—and lean on regression fixtures for image outputs when exact matching is required. Note manual verification steps in notebook markdown when automation is impractical.

## Commit & Pull Request Guidelines
Recent commits favor concise, sentence-case subjects (e.g., "Added qwen-image-edit 2509 benchmark..."). Keep subjects under 72 characters, add wrapped context paragraphs, and reference dataset or issue IDs. Pull requests should state intent, list commands executed (benchmarks, lint, tests), describe generated assets and storage paths, and include screenshots for visual changes.
