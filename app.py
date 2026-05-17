from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from src.predict import (
    DEFAULT_BATCH_PATH,
    MANUAL_INPUT_COLUMNS,
    export_predictions,
    predict_manual,
    score_batch_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_VALUES = {
    "dist_min_ci_0_5h": 5000.0,
    "alignment_abs": 0.5,
    "closing_speed_m_per_h": 100.0,
    "dt_first_last_0_5h": 4.0,
    "num_perimeters_0_5h": 3.0,
    "spread_bearing_cos": 0.25,
}

FIELD_HELP = {
    "dist_min_ci_0_5h": "Closest observed distance from the fire to the community interface in meters.",
    "alignment_abs": "How directly the fire spread aligns with the target area, from 0 to 1.",
    "closing_speed_m_per_h": "Estimated closing speed toward the interface in meters per hour.",
    "dt_first_last_0_5h": "Hours between the first and last perimeter observations in the 0-5h window.",
    "num_perimeters_0_5h": "Number of perimeter observations captured in the first 5 hours.",
    "spread_bearing_cos": "Cosine of spread direction. Values near 1 suggest aligned movement.",
}


def _build_manual_plot(results: pd.DataFrame):
    row = results.iloc[0]
    horizons = ["12h", "24h", "48h", "72h"]
    probs = [row["prob_12h"], row["prob_24h"], row["prob_48h"], row["prob_72h"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(horizons, probs, color=["#C85C34", "#D98E04", "#4C6A92", "#2B4C7E"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Threat probability")
    ax.set_title("Wildfire threat forecast")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")

    fig.tight_layout()
    return fig


def run_manual_prediction(*values):
    inputs = dict(zip(MANUAL_INPUT_COLUMNS, values))
    results = predict_manual(inputs)
    summary = (
        f"Estimated evacuation-zone threat: "
        f"12h {results.at[0, 'prob_12h']:.1%}, "
        f"24h {results.at[0, 'prob_24h']:.1%}, "
        f"48h {results.at[0, 'prob_48h']:.1%}, "
        f"72h {results.at[0, 'prob_72h']:.1%}."
    )
    display_df = pd.DataFrame(
        {
            "horizon": ["12 hours", "24 hours", "48 hours", "72 hours"],
            "threat_probability": [
                results.at[0, "prob_12h"],
                results.at[0, "prob_24h"],
                results.at[0, "prob_48h"],
                results.at[0, "prob_72h"],
            ],
        }
    )
    return summary, display_df, _build_manual_plot(results)


def run_batch_prediction(file_obj):
    file_path = file_obj.name if file_obj is not None else str(DEFAULT_BATCH_PATH)
    results = score_batch_file(file_path)
    download_path = export_predictions(results)
    summary = (
        f"Scored {len(results)} wildfire events. "
        f"Use the table preview below or download the full predictions CSV."
    )
    return summary, results, download_path


with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", secondary_hue="slate")) as demo:
    gr.Markdown(
        """
        # Wildfire Survival Forecast
        """
    )

    with gr.Tab("Single Event"):
        gr.Markdown(
            "Enter the six engineered-input fields used by the project model to estimate"
            " the probability that a wildfire threatens an evacuation zone within 12, 24, 48, and 72 hours."
        )

        manual_inputs = []
        with gr.Row():
            with gr.Column():
                for name in MANUAL_INPUT_COLUMNS[:3]:
                    manual_inputs.append(
                        gr.Number(value=DEFAULT_VALUES[name], label=name, info=FIELD_HELP[name])
                    )
            with gr.Column():
                for name in MANUAL_INPUT_COLUMNS[3:]:
                    manual_inputs.append(
                        gr.Number(value=DEFAULT_VALUES[name], label=name, info=FIELD_HELP[name])
                    )

        manual_button = gr.Button("Forecast Threat", variant="primary")
        manual_summary = gr.Textbox(label="Summary", interactive=False)
        manual_table = gr.Dataframe(label="Probability by Horizon", interactive=False)
        manual_plot = gr.Plot(label="Forecast Chart")

        manual_button.click(
            fn=run_manual_prediction,
            inputs=manual_inputs,
            outputs=[manual_summary, manual_table, manual_plot],
        )

    with gr.Tab("Batch CSV"):
        gr.Markdown(
            "Upload a processed CSV like `data/test_processed.csv`. If you leave this blank,"
            " the app will score the bundled test file."
        )
        batch_file = gr.File(label="Processed wildfire CSV", file_types=[".csv"])
        batch_button = gr.Button("Run Batch Forecast", variant="primary")
        batch_summary = gr.Textbox(label="Batch Summary", interactive=False)
        batch_table = gr.Dataframe(label="Predictions Preview", interactive=False)
        batch_download = gr.File(label="Download Predictions")

        batch_button.click(
            fn=run_batch_prediction,
            inputs=[batch_file],
            outputs=[batch_summary, batch_table, batch_download],
        )


if __name__ == "__main__":
    demo.launch()
