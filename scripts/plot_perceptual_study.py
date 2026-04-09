#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


METHOD_KEYS = ["dhoi", "textgraspdiff", "grabnet"]
METHOD_LABELS = ["DHOI", "TextGraspDiff", "GrabNet"]
METHOD_COLORS = ["#4C72B0", "#F28E2B", "#B7B7B7"]


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str(row["question"]).strip()
            counts = []
            for key in METHOD_KEYS:
                value = row.get(key, "0")
                counts.append(float(value) if value not in ("", None) else 0.0)
            rows.append((question, counts))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def plot_rows(rows, output_path, title=None):
    n_groups = len(rows)
    x = np.arange(n_groups, dtype=np.float32)
    width = 0.22

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    has_real_votes = False
    for method_idx, (label, color) in enumerate(zip(METHOD_LABELS, METHOD_COLORS)):
        values = []
        is_placeholder = []
        for _, counts in rows:
            total = sum(counts)
            if total > 0:
                has_real_votes = True
                values.append(100.0 * counts[method_idx] / total)
                is_placeholder.append(False)
            else:
                values.append(0.0)
                is_placeholder.append(True)

        bar_positions = x + (method_idx - 1) * width
        bars = ax.bar(
            bar_positions,
            values,
            width=width,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            label=label,
        )

        for bar, val, placeholder in zip(bars, values, is_placeholder):
            if placeholder:
                bar.set_facecolor("white")
                bar.set_hatch("//")
                bar.set_linewidth(1.5)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    2.0,
                    "TBD",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=color,
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + 1.2,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels([q for q, _ in rows])
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    if title:
        ax.set_title(title)
    elif not has_real_votes:
        ax.set_title("Perceptual Study Template")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Plot perceptual study results for DHOI")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="experiments/user_study/perceptual_study_results_template.csv",
        help="CSV with columns: question,dhoi,grabnet,textgraspdiff",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/user_study/perceptual_study_template.png",
        help="Output PNG path; a PDF with the same stem is also saved",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    plot_rows(rows, args.output, title=args.title)


if __name__ == "__main__":
    main()
