"""Build the combined bridge/final correctness-by-layer figure (3 rows: Base
(LoRA) / Delta (LEAP-LoRA) / Meta (LEAP), 2 columns: Bridge entity / Final
Answer) from two `main_logitlens.py` JSONL outputs — one run with
`--entity_type intermediate` (bridge) and one with `--entity_type final`
(final answer).
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 0


def finite_mean(values: Iterable[float]):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return mean(vals)


def bootstrap_mean_ci95(
    values: Iterable[float],
    rng: Any,
    *,
    num_samples: int = BOOTSTRAP_SAMPLES,
):
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))])
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        value = float(vals[0])
        return value, value

    sample_means = rng.choice(vals, size=(num_samples, vals.size), replace=True).mean(
        axis=1
    )
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return float(low), float(high)


def load_rows(logitlens_input: Path):
    """Read a `main_logitlens.py` JSONL (or a directory containing exactly
    one `*_logitlens_token_identity.jsonl`) into per-case summary rows."""
    if logitlens_input.is_file():
        jsonl_path = logitlens_input
    else:
        matches = sorted(logitlens_input.glob("*_logitlens_token_identity.jsonl"))
        if not matches:
            raise FileNotFoundError(
                f"No *_logitlens_token_identity.jsonl found under {logitlens_input}"
            )
        jsonl_path = matches[0]

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            base = rec["base"]
            meta = rec["meta"]
            base_lp = base["logprob_by_layer"]
            meta_lp = meta["logprob_by_layer"]
            if len(base_lp) != len(meta_lp):
                raise ValueError(
                    f"case_id={rec.get('case_id')}: base/meta layer count mismatch "
                    f"({len(base_lp)} vs {len(meta_lp)})"
                )
            rows.append(
                {
                    "case_id": rec.get("case_id"),
                    "entity": rec.get("entity"),
                    "base_correct": bool(base.get("correct")),
                    "meta_correct": bool(meta.get("correct")),
                    "correctness_group": correctness_group(
                        bool(base.get("correct")), bool(meta.get("correct"))
                    ),
                    "base_final_logprob_by_layer": base_lp,
                    "meta_final_logprob_by_layer": meta_lp,
                    "final_delta_logprob_by_layer": [
                        m - b for b, m in zip(base_lp, meta_lp)
                    ],
                }
            )
    return rows


def correctness_group(base_correct: bool, meta_correct: bool):
    base = "base_correct" if base_correct else "base_incorrect"
    meta = "meta_correct" if meta_correct else "meta_incorrect"
    return f"{base}__{meta}"


GROUP_STYLE = {
    "base_correct__meta_correct": ("LoRA correct / LEAP correct", "#2f6fbb", "o"),
    "base_incorrect__meta_correct": (
        "LoRA incorrect / LEAP correct",
        "#269c6b",
        "s",
    ),
    "base_correct__meta_incorrect": (
        "LoRA correct / LEAP incorrect",
        "#d9822b",
        "^",
    ),
    "base_incorrect__meta_incorrect": (
        "LoRA incorrect / LEAP incorrect",
        "#9b4dca",
        "D",
    ),
}
GROUP_ORDER = list(GROUP_STYLE)


def ordered_groups(groups: Dict[str, dict]):
    seen = set()
    for key in GROUP_ORDER:
        group = groups.get(key)
        if group is not None:
            seen.add(key)
            yield key, group
    for key in sorted(set(groups) - seen):
        yield key, groups[key]


def summarize_layers(rows: List[Dict[str, Any]], rng: Any, *, value_key: str):
    if not rows:
        return {"num_cases": 0, "by_layer": []}

    num_layers = len(rows[0][value_key])
    by_layer = []
    for layer_idx in range(num_layers):
        values = [row[value_key][layer_idx] for row in rows]
        ci_low, ci_high = bootstrap_mean_ci95(values, rng)
        by_layer.append(
            {
                "layer": layer_idx,
                "mean": finite_mean(values),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return {"num_cases": len(rows), "by_layer": by_layer}


def summarize_by_correctness(rows: List[Dict[str, Any]], rng: Any, *, value_key: str):
    summary: Dict[str, Any] = {"num_rows": len(rows), "by_correctness_group": {}}
    groups = {row["correctness_group"] for row in rows}
    for group in GROUP_ORDER + sorted(groups - set(GROUP_ORDER)):
        group_rows = [row for row in rows if row["correctness_group"] == group]
        summary["by_correctness_group"][group] = summarize_layers(
            group_rows, rng, value_key=value_key
        )
    return summary


def check_bridge_final_correctness_alignment(
    bridge_rows: List[Dict[str, Any]], final_rows: List[Dict[str, Any]]
):
    """Warn if the bridge/final runs disagree on which cases were
    correct/incorrect for base (LoRA) or meta (LEAP).

    The combined plot shows Bridge/Final side by side under a single shared
    per-group case count; that's only meaningful if both `main_logitlens.py`
    runs produced identical base/meta correctness labels per case_id (i.e.
    the runs were reproducible).
    """
    bridge_by_case = {row["case_id"]: row["correctness_group"] for row in bridge_rows}
    final_by_case = {row["case_id"]: row["correctness_group"] for row in final_rows}
    common = set(bridge_by_case) & set(final_by_case)
    mismatched = sorted(
        case_id
        for case_id in common
        if bridge_by_case[case_id] != final_by_case[case_id]
    )
    only_bridge = sorted(set(bridge_by_case) - set(final_by_case))
    only_final = sorted(set(final_by_case) - set(bridge_by_case))

    if mismatched or only_bridge or only_final:
        print(
            "WARNING: bridge/final correctness groups do not match "
            f"({len(mismatched)} case(s) with a differing group, "
            f"{len(only_bridge)} only in bridge, {len(only_final)} only in "
            "final). The combined plot's per-group case counts (n=...) may "
            "not describe both columns accurately."
        )
        if mismatched:
            print(f"  Mismatched case_ids (first 10): {mismatched[:10]}")
        return False
    return True


def draw_delta_logprob_by_layer(
    ax: Any,
    summary: Dict[str, Any],
    *,
    title: str,
    ylabel: str,
    ylimit_mode: str = "full_ci",
    show_legend: bool = True,
    show_xlabel: bool = True,
    ytick_interval: float = 1,
):
    from matplotlib.ticker import MultipleLocator

    groups = summary["by_correctness_group"]
    plotted = 0
    max_layer = 0
    mean_values_for_ylim = []

    for group_key, group in ordered_groups(groups):
        if not group or group.get("num_cases", 0) == 0:
            continue

        by_layer = group["by_layer"]
        layers = [int(row["layer"]) + 1 for row in by_layer]
        values = [row["mean"] for row in by_layer]
        for value in values:
            if value is not None and math.isfinite(float(value)):
                mean_values_for_ylim.append(float(value))
        lower = [row.get("ci95_low", value) for row, value in zip(by_layer, values)]
        upper = [row.get("ci95_high", value) for row, value in zip(by_layer, values)]
        max_layer = max(max_layer, max(layers))
        label, color, marker = GROUP_STYLE.get(
            group_key, (group_key.replace("__", " / "), None, "o")
        )

        ax.fill_between(layers, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            layers,
            values,
            marker=marker,
            markersize=4.0,
            markevery=2,
            linewidth=2.0,
            color=color,
            label=f"{label} (n={group['num_cases']})",
        )
        plotted += 1

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_title(title, fontsize=16, pad=8)
    ax.set_xlabel("Layer" if show_xlabel else "", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(1, max_layer if max_layer else 32)
    ticks = [1] + list(range(5, (max_layer or 32) + 1, 5))
    if ticks[-1] != (max_layer or 32):
        ticks.append(max_layer or 32)
    ax.set_xticks(ticks)
    if ylimit_mode == "mean" and mean_values_for_ylim:
        y_min = min(mean_values_for_ylim)
        y_max = max(mean_values_for_ylim)
        y_range = y_max - y_min
        y_pad = max(y_range * 0.12, 0.25)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    elif ylimit_mode != "full_ci":
        raise ValueError(f"Unknown ylimit_mode: {ylimit_mode}")
    ax.yaxis.set_major_locator(MultipleLocator(ytick_interval))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=11)

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if plotted and show_legend:
        ncol = 1 if plotted <= 2 else 2
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=ncol,
            frameon=False,
            fontsize=16,
            handlelength=2.4,
            columnspacing=1.2,
        )
    return handles, labels


def save_bridge_final_combined_plot(
    bridge_summaries: Dict[str, Dict[str, Any]],
    final_summaries: Dict[str, Dict[str, Any]],
    save_path: Path,
):
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 11.0), constrained_layout=True)
    panel_specs = [
        ("base", "Logprob (LoRA)", "full_ci", 2),
        ("delta", "Delta logprob (LEAP - LoRA)", "mean", 1),
        ("meta", "Logprob (LEAP)", "full_ci", 2),
    ]
    column_specs = [
        ("Bridge", bridge_summaries),
        ("Final Answer", final_summaries),
    ]

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    for row_idx, (key, ylabel, ylimit_mode, ytick_interval) in enumerate(panel_specs):
        for col_idx, (col_title, summaries) in enumerate(column_specs):
            handles, labels = draw_delta_logprob_by_layer(
                axes[row_idx][col_idx],
                summaries[key],
                title=col_title if row_idx == 0 else "",
                ylabel=ylabel if col_idx == 0 else "",
                ylimit_mode=ylimit_mode,
                show_legend=False,
                show_xlabel=row_idx == len(panel_specs) - 1,
                ytick_interval=ytick_interval,
            )
            if not legend_handles:
                legend_handles = handles
                legend_labels = labels

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
            fontsize=16.0,
            handlelength=2.4,
            columnspacing=1.4,
        )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summaries(rows: List[Dict[str, Any]], rng: Any):
    return {
        "base": summarize_by_correctness(
            rows, rng, value_key="base_final_logprob_by_layer"
        ),
        "delta": summarize_by_correctness(
            rows, rng, value_key="final_delta_logprob_by_layer"
        ),
        "meta": summarize_by_correctness(
            rows, rng, value_key="meta_final_logprob_by_layer"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the combined 3x2 (Base/Delta/Meta x Bridge/Final) "
            "correctness-by-layer figure from two main_logitlens.py JSONL "
            "outputs."
        )
    )
    parser.add_argument(
        "--bridge_input",
        type=Path,
        required=True,
        help="main_logitlens.py --entity_type intermediate output dir or jsonl.",
    )
    parser.add_argument(
        "--final_input",
        type=Path,
        required=True,
        help="main_logitlens.py --entity_type final output dir or jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults beside --bridge_input.",
    )
    parser.add_argument(
        "--plot_name", default="bridge_final_correctness_logprob_grid.png"
    )
    parser.add_argument("--summary_name", default="bridge_final_summary.json")
    args = parser.parse_args()

    bridge_dir = (
        args.bridge_input if args.bridge_input.is_dir() else args.bridge_input.parent
    )
    output_dir = args.output_dir or bridge_dir / "analysis_final_token_delta"
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_rows = load_rows(args.bridge_input)
    final_rows = load_rows(args.final_input)
    check_bridge_final_correctness_alignment(bridge_rows, final_rows)

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    bridge_summaries = build_summaries(bridge_rows, rng)
    final_summaries = build_summaries(final_rows, rng)

    summary_path = output_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"bridge": bridge_summaries, "final": final_summaries},
            fh,
            ensure_ascii=False,
            indent=2,
        )

    plot_path = output_dir / args.plot_name
    save_bridge_final_combined_plot(bridge_summaries, final_summaries, plot_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved combined plot: {plot_path}")


if __name__ == "__main__":
    main()
