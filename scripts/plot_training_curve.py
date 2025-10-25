# scripts/plot_training_curve.py
import os

import matplotlib.pyplot as plt
import pandas as pd

PATH = "artifacts/mix_epoch_metrics.csv"
OUT_LOSS = "artifacts/training_curve.png"
OUT_METRICS = "artifacts/metrics_curve_dual_axis.png"


def main():
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"Could not find {PATH}")

    df = pd.read_csv(PATH).sort_values("epoch")

    # Loss plot
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["eval_loss"], label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss (T5-Small MIX)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_LOSS, dpi=200, bbox_inches="tight")

    # Metrics plot with dual y-axes
    fig, ax_left = plt.subplots()
    # Left axis: ROUGE-1 and ROUGE-L
    if "eval_rouge1" in df.columns:
        ax_left.plot(df["epoch"], df["eval_rouge1"], label="ROUGE-1")
    if "eval_rougeL" in df.columns:
        ax_left.plot(df["epoch"], df["eval_rougeL"], label="ROUGE-L")
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("ROUGE")
    ax_left.set_title("Validation Metrics (T5-Small MIX)")

    # Right axis: BLEU
    ax_right = ax_left.twinx()
    if "eval_bleu" in df.columns:
        ax_right.plot(df["epoch"], df["eval_bleu"], label="BLEU")
    ax_right.set_ylabel("BLEU")

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    fig.tight_layout()
    fig.savefig(OUT_METRICS, dpi=200, bbox_inches="tight")

    print(f"Wrote {OUT_LOSS}")
    print(f"Wrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
