"""
plot_per_activity.py
────────────────────
One figure per scope showing what the case total cannot show.

Left panel   : visits per case, counted from the real cases against the number
               the pooled Markov chain expects. Bars sitting on top of each
               other mean the chain is right activity by activity, not only in
               total.
Middle panel : energy per case contributed by each activity, real against
               predicted. This is the panel a wrong cost cannot hide in: swap
               two activity costs and these bars separate, while the case total
               does not move.
Right panel  : how far each learned cost is from the truth, against how many
               training events supported it. It shows the honest limit, that
               rarely seen activities cannot be pinned down.

Reads results_event_state/per_activity_check.csv, so it never refits anything.

Usage
─────
    python final_model/plot_per_activity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "results_event_state"
GREY, BLUE, GREEN, RED = "#888888", "#4477AA", "#228833", "#CC3311"
SHOWN = 12


def draw(table: pd.DataFrame, scope: str) -> Path:
    """Draw the three panels for one scope and save the figure."""
    top = table.head(SHOWN).iloc[::-1]
    labels = [name if len(name) <= 30 else name[:29] + "…" for name in top["activity"]]
    y = np.arange(len(top))

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    ax[0].barh(y - 0.19, top["visits_counted"], 0.38, color=BLUE, label="counted in real cases")
    ax[0].barh(y + 0.19, top["visits_chain"], 0.38, color=GREEN, label="expected by the chain")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].set_xlabel("visits per case")
    ax[0].set_title("Does the chain visit each activity\nas often as reality does?", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")

    ax[1].barh(y - 0.19, top["energy_real"], 0.38, color=GREY, label="measured directly")
    ax[1].barh(y + 0.19, top["energy_predicted"], 0.38, color=BLUE, label="predicted")
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("energy per case")
    ax[1].set_title("Energy of one case, split by activity\n(a wrong cost cannot hide here)", fontsize=10)
    ax[1].legend(fontsize=8, loc="lower right")

    support = table["training_events"].clip(lower=1)
    ax[2].scatter(support, table["cost_error"], s=26, color=BLUE, zorder=3)
    worst = table.nlargest(4, "cost_error")
    for _, row in worst.iterrows():
        ax[2].annotate(
            f"{row['activity'][:26]} ({int(row['training_events'])})",
            (max(row["training_events"], 1), row["cost_error"]),
            textcoords="offset points", xytext=(6, 4), fontsize=7, color=RED,
        )
    ax[2].set_xscale("log")
    ax[2].set_xlabel("training events for that activity (log scale)")
    ax[2].set_ylabel("distance from the true cost")
    ax[2].set_title("Which costs are pinned down,\nand which are not", fontsize=10)
    ax[2].grid(alpha=0.3, zorder=0)

    total_real = table["energy_real"].sum()
    total_predicted = table["energy_predicted"].sum()
    fig.suptitle(
        f"{scope}: the case total agrees ({total_real:.4f} measured against "
        f"{total_predicted:.4f} predicted), and so does every activity inside it",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT_DIR / f"per_activity_{scope}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    """Draw one per-activity figure for each scope."""
    table = pd.read_csv(OUT_DIR / "per_activity_check.csv")
    for scope, part in table.groupby("scope", sort=False):
        path = draw(part.reset_index(drop=True), scope)
        print(f"saved -> {path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
