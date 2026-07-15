"""CLI benchmark runner for Qwen-Image-Edit-2509."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable

import torch
from PIL import Image

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError as exc:
    raise RuntimeError("diffusers>=0.27.2 is required. Run `uv sync` or `pip install diffusers>=0.27.2`.") from exc

try:
    import polars as pl
except ImportError as exc:  # pragma: no cover - import guard for missing optional deps
    raise RuntimeError(
        "Polars is required for the benchmark. Install dependencies with `uv sync` or "
        "`pip install polars`."
    ) from exc


def resolve_image(path_like: str) -> Image.Image:
    """Load an image from the provided path or fall back to the ./data directory."""
    candidate_paths: Iterable[Path] = (
        Path(path_like),
        Path("data") / path_like,
    )
    for candidate in candidate_paths:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    raise FileNotFoundError(
        f"Unable to locate {path_like}. Place the file in the working directory or ./data/."
    )


def build_inputs(image1: Image.Image, image2: Image.Image, prompt: str) -> Dict[str, object]:
    """Prepare the pipeline input payload with deterministic seeding."""
    return {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2509 Inference Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen-Image-Edit-2509",
        help="Model ID or local path to load with diffusers.",
    )
    parser.add_argument(
        "--input1-path", type=str, default="input1.png", help="Path to the first input image."
    )
    parser.add_argument(
        "--input2-path", type=str, default="input2.png", help="Path to the second input image."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "The magician bear is on the left, the alchemist bear is on the right, facing each "
            "other in the central park square."
        ),
        help="Prompt passed to the image-edit pipeline.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for benchmark metrics and generated assets.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output_image_edit_plus.png",
        help="Filename for the generated image (saved inside --results-dir).",
    )
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of inference runs for the benchmark."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not detected. Ensure you are running on an NVIDIA GPU.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = results_dir / args.output_name
    metrics_path = results_dir / "qwen_image_edit_benchmark.parquet"
    summary_path = results_dir / "qwen_image_edit_benchmark_summary.csv"

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    device_name = torch.cuda.get_device_name(0)
    print(f"Pipeline {args.model_path} loaded on {device_name}")

    image1 = resolve_image(args.input1_path)
    image2 = resolve_image(args.input2_path)
    print("Input images loaded")

    inputs = build_inputs(image1, image2, args.prompt)

    timing_rows = []
    print(f"Running inference {args.num_runs} times on {device_name}...")

    for run_idx in range(1, args.num_runs + 1):
        start_time = perf_counter()
        with torch.inference_mode():
            output = pipeline(**inputs)
        elapsed_time = perf_counter() - start_time
        timing_rows.append({"run": run_idx, "inference_time_seconds": elapsed_time})
        print(f"Run {run_idx}: {elapsed_time:.2f} seconds")

    df = pl.DataFrame(timing_rows)
    summary = df.select(
        [
            pl.col("inference_time_seconds").mean().alias("mean_seconds"),
            pl.col("inference_time_seconds").std().alias("std_seconds"),
            pl.col("inference_time_seconds").min().alias("min_seconds"),
            pl.col("inference_time_seconds").max().alias("max_seconds"),
        ]
    )

    print("\nBenchmark summary:")
    print(summary)

    df.write_parquet(metrics_path)
    summary.write_csv(summary_path)
    print(f"Timing metrics saved to {metrics_path.resolve()}")
    print(f"Summary metrics saved to {summary_path.resolve()}")

    output_image = output.images[0]
    output_image.save(output_image_path)
    print(f"Output image saved at {output_image_path.resolve()}")


if __name__ == "__main__":
    main()
