"""Figures for the report.

Chart choices follow one rule each: magnitude comparisons are bars on a shared
axis, the compute figures are separate panels rather than a dual axis (train
time and parameter count do not share a scale), and the temperature sweep is a
line chart because temperature is continuous.  Every figure is direct-labelled
so it stays readable in greyscale and in print, and the full numbers are always
available as a table in ``results/metrics.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

SHORT = {
    "Transformer (scratch)": "Transformer\n(scratch)",
    "mT5-small (fine-tuned)": "mT5-small\n(fine-tuned)",
    "Llama-3.2-1B (LoRA)": "Llama-3.2-1B\n(LoRA)",
    "Gemma-2-2B (LoRA)": "Gemma-2-2B\n(LoRA)",
    "Llama-3.2-1B (3-shot prompt)": "Llama-3.2-1B\n(3-shot)",
    "Gemma-2-2B (3-shot prompt)": "Gemma-2-2B\n(3-shot)",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 150,
    })


def _ax_grid(ax, log: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    if log:
        ax.set_yscale("log")


def _labels(rows: list[dict]) -> list[str]:
    return [SHORT.get(r["model"], r["model"]) for r in rows]


def _save(fig, name: str) -> Path:
    C.ensure_dirs()
    path = C.PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path.relative_to(C.ROOT)}")
    return path


def _annotate_bars(ax, bars, fmt="{:.1f}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK_SOFT)


# --------------------------------------------------------------------------- #

def plot_quality(rows: list[dict]) -> Path:
    """Grouped bars: BLEU and chrF++ side by side for every system."""
    _style()
    x = range(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2.5, 4.2))
    b1 = ax.bar([i - width / 2 - 0.01 for i in x], [r["bleu"] for r in rows],
                width, label="BLEU", color=SERIES[0], linewidth=0)
    b2 = ax.bar([i + width / 2 + 0.01 for i in x], [r["chrf"] for r in rows],
                width, label="chrF++", color=SERIES[1], linewidth=0)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("score")
    ax.set_title(f"Translation quality on the held-out test set "
                 f"({rows[0]['test_sentences']} sentences, {C.SRC_LANG}→{C.TGT_LANG})")
    ax.legend(loc="upper left", ncols=2)
    _ax_grid(ax)
    return _save(fig, "quality_bleu_chrf.png")


def plot_compute(rows: list[dict]) -> Path:
    """Three panels: size, adaptation cost and inference cost.

    Deliberately not one chart with two y-axes - these quantities share no scale.
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.4))
    labels = _labels(rows)
    x = range(len(rows))

    panels = [
        ("total_params_m", "Model size (million parameters)", True, "{:.0f}"),
        ("train_time_s", "Training / fine-tuning time (s)", False, "{:.0f}"),
        ("ms_per_sentence", "Inference cost (ms per sentence)", False, "{:.0f}"),
    ]
    for ax, (key, title, log, fmt) in zip(axes, panels):
        bars = ax.bar(list(x), [r[key] for r in rows], 0.6,
                      color=SERIES[0], linewidth=0)
        _annotate_bars(ax, bars, fmt)
        ax.set_xticks(list(x))
        # Six two-line labels do not fit a 4.6in panel horizontally.
        rotation = 30 if len(rows) > 4 else 0
        ax.set_xticklabels(labels, rotation=rotation,
                           ha="right" if rotation else "center")
        ax.set_title(title)
        _ax_grid(ax, log=log)
        if log:
            ax.set_ylabel("parameters (M, log scale)")
    return _save(fig, "compute_cost.png")


def plot_quality_vs_size(rows: list[dict]) -> Path:
    """Is quality bought with parameters? Colour encodes architecture."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    arch_colour = {"encoder-decoder": SERIES[0], "decoder-only": SERIES[1]}
    seen = set()
    for r in rows:
        arch = r["architecture"]
        ax.scatter(r["total_params_m"], r["chrf"], s=90,
                   color=arch_colour.get(arch, SERIES[2]),
                   edgecolor=SURFACE, linewidth=2,
                   label=arch if arch not in seen else None, zorder=3)
        seen.add(arch)
        ax.annotate(r["model"], (r["total_params_m"], r["chrf"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=INK_SOFT)
    ax.set_xscale("log")
    # Room for the point labels, which extend to the right of each marker.
    ax.set_xmargin(0.3)
    ax.margins(y=0.15)
    ax.set_xlabel("parameters (M, log scale)")
    ax.set_ylabel("chrF++")
    ax.set_title("Quality against model size")
    # Upper left: the point labels extend rightwards, so a right-hand legend
    # collides with the lowest-scoring system's label.
    ax.legend(loc="upper left")
    _ax_grid(ax)
    return _save(fig, "quality_vs_size.png")


def plot_temperature(sweep: list[dict]) -> Path:
    """BLEU and chrF++ as a function of decoding temperature."""
    _style()
    models = []
    for r in sweep:
        if r["model"] not in models:
            models.append(r["model"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for metric, ax, title in (("bleu", axes[0], "BLEU vs sampling temperature"),
                              ("chrf", axes[1], "chrF++ vs sampling temperature")):
        for i, model in enumerate(models):
            pts = sorted((r for r in sweep if r["model"] == model),
                         key=lambda r: r["temperature"])
            xs = [r["temperature"] for r in pts]
            ys = [r[metric] for r in pts]
            colour = SERIES[i % len(SERIES)]
            ax.plot(xs, ys, color=colour, linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=model, zorder=3)
            # Direct label at the right-hand end keeps identity off colour alone.
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=8, color=INK_SOFT, va="center")
        ax.set_xlabel("temperature (0.0 = greedy)")
        ax.set_ylabel(metric.upper() if metric == "bleu" else "chrF++")
        ax.set_title(title)
        _ax_grid(ax)
    axes[0].legend(loc="lower left")
    return _save(fig, "temperature_sweep.png")


def plot_loss_curves(histories: dict[str, list[dict]]) -> Path | None:
    """Training and validation loss per epoch for the two trained seq2seq systems."""
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return None
    _style()
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (name, hist) in zip(axes[0], histories.items()):
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], color=SERIES[0],
                linewidth=2, marker="o", markersize=4, label="train loss")
        ax.plot(epochs, [h["val_loss"] for h in hist], color=SERIES[1],
                linewidth=2, marker="o", markersize=4, label="validation loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.set_title(name)
        ax.legend(loc="upper right")
        _ax_grid(ax)
    return _save(fig, "loss_curves.png")


def plot_all(rows: list[dict], sweep: list[dict],
             histories: dict[str, list[dict]]) -> list[Path]:
    made = [plot_quality(rows), plot_compute(rows), plot_quality_vs_size(rows)]
    if sweep:
        made.append(plot_temperature(sweep))
    curves = plot_loss_curves(histories)
    if curves:
        made.append(curves)
    return made
