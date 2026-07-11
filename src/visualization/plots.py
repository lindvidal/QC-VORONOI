import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple

from ..config import FIGURES_DIR, FIGURE_DPI


def plot_annual_rainfall_histogram(
    df_events: pd.DataFrame,
    mit_minutes: int = 30,
    bins: int = 50,
    save: bool = True,
    filename: str = "annual_rainfall_distribution",
) -> plt.Figure:
    df = df_events[df_events["mit_minutes"] == mit_minutes].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["yearly_rainfall"], bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Annual Rainfall (mm)")
    ax.set_ylabel("Number of Stations")
    ax.set_title(f"Distribution of Annual Rainfall (MIT={mit_minutes}min)")
    ax.grid(True, alpha=0.3)
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_event_properties_boxplot(
    df_events: pd.DataFrame,
    mit_minutes: int = 30,
    save: bool = True,
    filename: str = "event_properties_boxplot",
) -> plt.Figure:
    df = df_events[df_events["mit_minutes"] == mit_minutes].copy()
    props = ["mean_rainfall_depth", "mean_rainfall_duration", "mean_rainfall_intensity", "mean_dry_time"]
    labels = ["Depth (mm)", "Duration (h)", "Intensity (mm/h)", "Dry Time (h)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (prop, label) in enumerate(zip(props, labels)):
        axes[i].boxplot(df[prop].dropna(), vert=True, patch_artist=True)
        axes[i].set_ylabel(label)
        axes[i].set_title(f"{label} Distribution")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle(f"Rainfall Event Properties (MIT={mit_minutes}min)", fontsize=14)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_qc_f1_heatmap(
    df_results: pd.DataFrame,
    mit_minutes: int,
    save: bool = True,
    filename: Optional[str] = None,
) -> plt.Figure:
    if filename is None:
        filename = f"f1_heatmap_mit{mit_minutes}"
    if df_results.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig
    pivot = df_results.pivot_table(index="p2", columns="p1", values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("p1 (upper tolerance)")
    ax.set_ylabel("p2 (lower tolerance)")
    ax.set_title(f"F1-Score Heatmap (MIT={mit_minutes}min)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.mean() else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="F1-Score")
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm_values: Tuple[int, int, int, int],
    title: str = "Confusion Matrix",
    save: bool = True,
    filename: str = "confusion_matrix",
) -> plt.Figure:
    tp, fp, tn, fn = cm_values
    matrix = np.array([[tp, fp], [fn, tn]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted HQ", "Predicted LQ"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True HQ", "True LQ"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_mit_comparison(
    df_events: pd.DataFrame,
    save: bool = True,
    filename: str = "mit_comparison",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    props = [
        ("yearly_rainfall", "Annual Rainfall (mm)"),
        ("rainfall_events", "Number of Events"),
        ("mean_rainfall_intensity", "Mean Intensity (mm/h)"),
        ("mean_rainfall_duration", "Mean Duration (h)"),
    ]
    for idx, (prop, ylabel) in enumerate(props):
        ax = axes[idx]
        for mit in sorted(df_events["mit_minutes"].unique()):
            data = df_events[df_events["mit_minutes"] == mit][prop].dropna()
            ax.hist(data, bins=30, alpha=0.5, label=f"MIT={mit}min", edgecolor="black")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{ylabel} by MIT")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_station_map(
    gauge_info: pd.DataFrame,
    quality_labels: Optional[pd.DataFrame] = None,
    save: bool = True,
    filename: str = "station_map",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))
    if quality_labels is not None:
        merged = gauge_info.merge(quality_labels, on="gauge_code", how="left")
        hq = merged[merged["quality"] == "HQ"]
        lq = merged[merged["quality"] == "LQ"]
        unk = merged[merged["quality"].isna()]
        if not hq.empty:
            ax.scatter(hq["long"], hq["lat"], c="green", s=20, alpha=0.6, label="HQ", edgecolors="black", linewidth=0.3)
        if not lq.empty:
            ax.scatter(lq["long"], lq["lat"], c="red", s=20, alpha=0.6, label="LQ", edgecolors="black", linewidth=0.3)
        if not unk.empty:
            ax.scatter(unk["long"], unk["lat"], c="gray", s=10, alpha=0.4, label="Unknown", edgecolors="black", linewidth=0.3)
    else:
        ax.scatter(gauge_info["long"], gauge_info["lat"], c="blue", s=10, alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Rainfall Station Map")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_f1_curve(
    df_results: pd.DataFrame,
    p1_values: List[float],
    p2_values: List[float],
    mit: int,
    save: bool = True,
    filename: Optional[str] = None,
) -> plt.Figure:
    if filename is None:
        filename = f"f1_curve_mit{mit}"
    fig, ax = plt.subplots(figsize=(12, 6))
    for p1 in p1_values:
        subset = df_results[df_results["p1"] == p1]
        ax.plot(subset["p2"], subset["f1"], marker="o", label=f"p1={p1:.2f}")
    ax.set_xlabel("p2 (lower tolerance)")
    ax.set_ylabel("F1-Score")
    ax.set_title(f"F1-Score vs p2 for different p1 values (MIT={mit}min)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    if not df_results.empty:
        best_idx = df_results["f1"].idxmax()
        best = df_results.loc[best_idx]
        ax.annotate(
            f"Best: p1={best['p1']:.2f}, p2={best['p2']:.2f}, F1={best['f1']:.3f}",
            xy=(best["p2"], best["f1"]),
            xytext=(best["p2"] + 0.05, best["f1"] - 0.05),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    return fig
