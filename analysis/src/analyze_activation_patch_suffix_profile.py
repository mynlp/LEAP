"""Build the combined activation-patch suffix-profile figure (two panels:
"LoRA to LEAP" / "LEAP to LoRA") from two `*_activation_patch_*.jsonl` files
produced by `src/main_activation_patch_lora.py` (one `--patch_direction
base_to_meta` run, one `meta_to_base` run).

Each row already carries `delta_logprob_by_distance[layer][distance]`
(distance 0 = final prompt token), so this script only needs to average that
across cases per (layer, distance) cell and plot the two directions side by
side.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DPI = 300
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 18
COLORBAR_LABEL_SIZE = 24
TITLE_SIZE = 24
VMAX_PERCENTILE = 98.0


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def collect_suffix_profile(jsonl_path: Path, *, suffix_tokens: int):
    records = load_jsonl(jsonl_path)
    sums: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None

    for rec in records:
        delta = rec.get("delta_logprob_by_distance")
        if not delta:
            continue
        num_layers = len(delta)
        if sums is None:
            sums = np.zeros((num_layers, suffix_tokens), dtype=np.float64)
            counts = np.zeros((num_layers, suffix_tokens), dtype=np.int64)
        elif sums.shape[0] != num_layers:
            continue

        for li, row in enumerate(delta):
            for di in range(min(suffix_tokens, len(row))):
                v = float(row[di])
                if math.isfinite(v):
                    sums[li, di] += v
                    counts[li, di] += 1

    if sums is None or counts is None:
        raise ValueError(f"No usable activation patch records found in {jsonl_path}")

    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def plot_title(jsonl_path: Path):
    if "base_to_meta" in jsonl_path.name:
        return "LoRA to LEAP"
    if "meta_to_base" in jsonl_path.name:
        return "LEAP to LoRA"
    return jsonl_path.stem


def symmetric_vmax(profiles: List[np.ndarray]):
    vals = [np.abs(p[np.isfinite(p)]) for p in profiles]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    return float(np.nanpercentile(np.concatenate(vals), VMAX_PERCENTILE)) or None


def x_ticks(num_layers: int):
    one_indexed = [1] + list(range(5, num_layers + 1, 5))
    if one_indexed[-1] != num_layers:
        one_indexed.append(num_layers)
    return np.asarray([idx - 1 for idx in one_indexed], dtype=int)


def draw_panel(
    ax: Any, mean_matrix: np.ndarray, *, title: str, norm: Any, show_ylabel: bool
):
    im = ax.imshow(
        mean_matrix.T,
        aspect="auto",
        cmap="RdBu_r",
        origin="lower",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel("Layer", fontsize=AXIS_LABEL_SIZE)
    if show_ylabel:
        ax.set_ylabel("Distance from final token", fontsize=TITLE_SIZE)

    xt = x_ticks(mean_matrix.shape[0])
    yt = np.arange(0, mean_matrix.shape[1], 5)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(idx + 1) for idx in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([str(idx) for idx in yt])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return im


def plot_pair(jsonl_paths: List[Path], profiles: List[np.ndarray], *, save_path: Path):
    vmax = symmetric_vmax(profiles)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax else None

    max_layers = max(p.shape[0] for p in profiles)
    max_suffix = max(p.shape[1] for p in profiles)
    panel_width = max(6.4, max_layers * 0.20)
    panel_height = max(4.2, max_suffix * 0.30)
    fig, axes = plt.subplots(
        1, 2, figsize=(panel_width * 2, panel_height), constrained_layout=True
    )

    im = None
    for idx, (ax, jsonl_path, mean) in enumerate(zip(axes, jsonl_paths, profiles)):
        im = draw_panel(
            ax, mean, title=plot_title(jsonl_path), norm=norm, show_ylabel=idx == 0
        )

    cbar = fig.colorbar(
        im,
        ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.08,
        pad=0.12,
        shrink=0.82,
        aspect=35,
    )
    cbar.set_label(
        "Change in the log probability of the final answer",
        fontsize=COLORBAR_LABEL_SIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot paired activation-patch suffix-profile heatmaps."
    )
    parser.add_argument("--jsonl", type=Path, nargs=2, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <first jsonl parent>/analysis_suffix_profile_pair.",
    )
    parser.add_argument("--output_name", default="paired_suffix_profile_heatmaps.png")
    parser.add_argument("--suffix_tokens", type=int, default=16)
    args = parser.parse_args()

    if args.suffix_tokens <= 0:
        raise ValueError("--suffix_tokens must be positive")

    output_dir = (
        args.output_dir or args.jsonl[0].parent / "analysis_suffix_profile_pair"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [
        collect_suffix_profile(path, suffix_tokens=args.suffix_tokens)
        for path in args.jsonl
    ]
    save_path = output_dir / args.output_name
    plot_pair(args.jsonl, profiles, save_path=save_path)
    print(f"Saved paired heatmap: {save_path}")


if __name__ == "__main__":
    main()
