import json
import sys
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np


def plot_time_per_token(json_file: str, output_image: str) -> None:
    """
    Creates a visualization of time per input token vs number of requests.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing benchmark data.
    output_image : str
        Path for saving the output figure.
    """
    # Basic style settings
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5

    # Load and process data
    with open(json_file, "r") as f:
        data: Dict = json.load(f)

    description_subtitle: str = data["metadata"]["parameters"]["description"]

    # Extract number of requests from results keys
    request_counts: List[int] = sorted(map(int, data["results"].keys()))

    # Initialize data containers - one list per repetition
    repetition_time_per_token: Dict[str, List[float]] = {}
    repetition_time_to_second_token: Dict[str, List[float]] = {}
    repetition_durations: Dict[str, List[List[float]]] = {}

    # Extract data from results and organize by repetition
    for num_requests in request_counts:
        entry = data["results"][str(num_requests)]

        for repetition_key in entry.keys():
            if repetition_key.startswith("repetition_"):
                summary = entry[repetition_key]["summary"]
                total_time = summary["total_time_seconds"]
                total_input_tokens = summary["total_input_tokens"]
                total_requests = summary["total_requests"]

                # Calculate time per input token (in seconds per token)
                time_per_token = total_time / total_input_tokens

                # Calculate time to second token (total time per request)
                time_to_second_token = total_time / total_requests

                # Extract individual request durations
                timing_data = entry[repetition_key].get("timing", {})
                request_durations = timing_data.get("request_duration_list", [])

                # Initialize repetition lists if not exists
                if repetition_key not in repetition_time_per_token:
                    repetition_time_per_token[repetition_key] = []
                    repetition_time_to_second_token[repetition_key] = []
                    repetition_durations[repetition_key] = []

                repetition_time_per_token[repetition_key].append(time_per_token)
                repetition_time_to_second_token[repetition_key].append(
                    time_to_second_token
                )
                repetition_durations[repetition_key].append(request_durations)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))

    # Style parameters - different colors for different repetitions
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Plot line positions
    x_positions = range(len(request_counts))

    # Plot 1: Time per Input Token
    for i, (repetition_key, time_values) in enumerate(
        repetition_time_per_token.items()
    ):
        color = colors[i % len(colors)]

        ax1.plot(
            x_positions,
            time_values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            label=f"Repetition {repetition_key.split('_')[1]}",
        )

        # Add value annotations for each repetition
        for x, value in zip(x_positions, time_values):
            ax1.annotate(
                f"{value:.2e}",
                xy=(x, value),
                xytext=(0, 10 + i * 5),  # Offset annotations for different repetitions
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    # Plot 2: Time to Second Token
    for i, (repetition_key, time_values) in enumerate(
        repetition_time_to_second_token.items()
    ):
        color = colors[i % len(colors)]

        ax2.plot(
            x_positions,
            time_values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            label=f"Repetition {repetition_key.split('_')[1]}",
        )

        # Add value annotations for each repetition
        for x, value in zip(x_positions, time_values):
            ax2.annotate(
                f"{value:.2f}",
                xy=(x, value),
                xytext=(0, 10 + i * 5),  # Offset annotations for different repetitions
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    # Styling for subplot 1
    ax1.set_title(
        "Time per Input Token vs. Number of Requests",
        fontsize=12,
        fontweight="normal",
    )
    ax1.set_ylabel("Time per Input Token (seconds)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(request_counts)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=10)
    ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Styling for subplot 2
    ax2.set_title(
        "Time to Second Token vs. Number of Requests",
        fontsize=12,
        fontweight="normal",
    )
    ax2.set_ylabel("Time to Second Token (seconds)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(request_counts)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=10)

    # Plot 3: Individual Request Durations
    for i, (repetition_key, duration_lists) in enumerate(repetition_durations.items()):
        color = colors[i % len(colors)]

        for request_idx, durations in enumerate(duration_lists):
            if durations:  # Check if durations list is not empty
                # Create x-values for this request count
                x_vals = [request_idx for _ in durations]

                ax3.scatter(
                    x_vals,
                    durations,
                    color=color,
                    s=15,  # Small circle size
                    alpha=0.4,  # Transparency
                    label=(
                        f"Repetition {repetition_key.split('_')[1]}"
                        if request_idx == 0
                        else ""
                    ),
                )

    # Styling for subplot 3
    ax3.set_title(
        "Individual Request Durations vs. Number of Requests",
        fontsize=12,
        fontweight="normal",
    )
    ax3.set_xlabel("Number of Requests", fontsize=12)
    ax3.set_ylabel("Request Duration (seconds)", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(request_counts)
    ax3.legend(fontsize=10)
    ax3.tick_params(labelsize=10)

    # Main title
    fig.suptitle(description_subtitle, fontsize=16, fontweight="bold")

    # Save with high quality
    plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Plot saved to: {output_image}")
    print("\nData Summary:")
    print("\n--- Time per Input Token ---")
    for repetition_key, time_values in repetition_time_per_token.items():
        print(f"  {repetition_key}:")
        for i, (requests, value) in enumerate(zip(request_counts, time_values)):
            print(f"    {requests} requests: {value:.2e} seconds per token")

    print("\n--- Time to Second Token ---")
    for repetition_key, time_values in repetition_time_to_second_token.items():
        print(f"  {repetition_key}:")
        for i, (requests, value) in enumerate(zip(request_counts, time_values)):
            print(f"    {requests} requests: {value:.2f} seconds per request")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_TTSC.py <json_file> <output_image>")
        sys.exit(1)

    json_file = sys.argv[1]
    output_image = sys.argv[2]
    plot_time_per_token(json_file, output_image)
