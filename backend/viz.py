"""Shared artifact-visual helpers (PNG overlays under artifacts/{verdict_id}/)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

_BG = "#08080A"
_FG = "#A1A1AA"
_ACCENT = "#58A6FF"
_DANGER = "#F85149"


def _style(ax):
    ax.set_facecolor("#121215")
    for spine in ax.spines.values():
        spine.set_color("#2A2A35")
    ax.tick_params(colors=_FG, labelsize=7)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color("#FFFFFF")
    ax.grid(color="#2A2A35", linewidth=0.4, alpha=0.6)


def save_heatmap(arr: np.ndarray, path, size=None):
    """Save a normalized 2-D array as an inferno heatmap PNG (no axes)."""
    a = arr.astype(np.float64)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    cmap = plt.get_cmap("inferno")
    rgba = (cmap(a) * 255).astype(np.uint8)
    img = Image.fromarray(rgba[:, :, :3])
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    img.save(path, "PNG")


def save_curve(x, y, path, title, xlabel, ylabel, marks=None):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
    fig.patch.set_facecolor(_BG)
    _style(ax)
    ax.plot(x, y, color=_ACCENT, linewidth=1.4)
    if marks:
        for mx in marks:
            ax.axvline(mx, color=_DANGER, linewidth=1.0, linestyle="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)


def save_spectrum_plot(freqs, profile, env_mean, env_std, peaks, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=110)
    fig.patch.set_facecolor(_BG)
    _style(ax)
    ax.fill_between(freqs, env_mean - 3 * env_std, env_mean + 3 * env_std,
                    color=_ACCENT, alpha=0.18, label="real envelope ±3σ")
    ax.plot(freqs, env_mean, color=_ACCENT, linewidth=0.8, alpha=0.7)
    ax.plot(freqs, profile, color="#FFFFFF", linewidth=1.3, label="this image")
    for f in peaks:
        ax.axvline(f, color=_DANGER, linewidth=1.0, linestyle="--")
    ax.set_title("FFT radial power spectrum vs calibrated real envelope", fontsize=9)
    ax.set_xlabel("normalized radial frequency", fontsize=8)
    ax.set_ylabel("log10 power", fontsize=8)
    leg = ax.legend(fontsize=7, facecolor="#121215", edgecolor="#2A2A35")
    for t in leg.get_texts():
        t.set_color(_FG)
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)


def save_bar_panel(labels, values, path, title, ref_lines=None, ylabel=""):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
    fig.patch.set_facecolor(_BG)
    _style(ax)
    ax.bar(labels, values, color=_ACCENT, width=0.6)
    if ref_lines:
        for label, val, color in ref_lines:
            ax.axhline(val, color=color, linewidth=1.0, linestyle="--", label=label)
        leg = ax.legend(fontsize=7, facecolor="#121215", edgecolor="#2A2A35")
        for t in leg.get_texts():
            t.set_color(_FG)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)
