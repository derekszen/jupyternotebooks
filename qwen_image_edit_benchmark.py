import os
import time
import torch
import argparse
import polars as pl
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2509 Inference Benchmark")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen-Image-Edit-2509",
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--input1_path", type=str, default="input1.png", help="Path to the first input image"
    )
    parser.add_argument(
        "--input2_path", type=str, default="input2.png", help="Path to the second input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
        help="Prompt for image editing",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_image_edit_plus.png",
        help="Path to save the output image",
    )
    parser.add_argument(
        "--num_runs", type=int, default=5, help="Number of inference runs for benchmarking"
    )
    args = parser.parse_args()

    # Load pipeline
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    print("Pipeline loaded on CUDA")

    # Load images
    image1 = Image.open(args.input1_path)
    image2 = Image.open(args.input2_path)
    print("Input images loaded")

    # Prepare inputs
    inputs = {
        "image": [image1, image2],
        "prompt": args.prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    # Benchmark inference
    timing_results = []
    print(f"Running inference {args.num_runs} times...")

    for i in range(args.num_runs):
        start_time = time.time()
        with torch.inference_mode():
            output = pipeline(**inputs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        timing_results.append(elapsed_time)
        print(f"Run {i + 1}: {elapsed_time:.2f} seconds")

    # Collect metrics
    df = pl.DataFrame(
        {"run": range(1, args.num_runs + 1), "inference_time_seconds": timing_results}
    )

    # Calculate statistics
    mean_time = df["inference_time_seconds"].mean()
    std_time = df["inference_time_seconds"].std()
    min_time = df["inference_time_seconds"].min()
    max_time = df["inference_time_seconds"].max()

    print("\nBenchmark Results:")
    print(f"Mean inference time: {mean_time:.2f} seconds")
    print(f"Standard deviation: {std_time:.2f} seconds")
    print(f"Min inference time: {min_time:.2f} seconds")
    print(f"Max inference time: {max_time:.2f} seconds")

    # Save metrics
    metrics_path = "benchmark_results.csv"
    df.write_csv(metrics_path)
    print(f"\nDetailed results saved to {os.path.abspath(metrics_path)}")

    # Save output image from last run
    output_image = output.images[0]
    output_image.save(args.output_path)
    print(f"Output image saved at {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
