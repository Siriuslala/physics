"""Visualize metrics from a local offline Weights & Biases run directory.

This script is designed to work for arbitrary offline W&B runs. It parses the
``run-*.wandb`` event file, discovers scalar metrics automatically, groups them
by metric namespace, and writes generic Matplotlib visualizations to an output
directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


RESERVED_X_COLUMNS = {"_step", "_runtime", "_timestamp"}
LAMBDA_EFF_WIDTH_COLOR = "#5BC5DB"
LAMBDA_EFF_HEIGHT_COLOR = "#9BC750"
LAMBDA_EFF_AXIS_LABEL_FONT_SIZE = 20
LAMBDA_EFF_TICK_FONT_SIZE = 20
LAMBDA_EFF_LEGEND_FONT_SIZE = 16


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize metrics from a local offline W&B run directory."
    )
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        required=True,
        help="Path to an offline W&B run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to save exported metrics and figures.",
    )
    parser.add_argument(
        "--max-metrics-per-figure",
        type=int,
        default=8,
        help="Maximum number of metrics plotted in a single figure.",
    )
    parser.add_argument(
        "--top-k-overview",
        type=int,
        default=24,
        help="Maximum number of high-variance metrics rendered in the overview figure.",
    )
    parser.add_argument(
        "--min-non-null-fraction",
        type=float,
        default=0.05,
        help="Minimum non-null fraction required for a metric to be plotted.",
    )
    return parser.parse_args()


def _normalize_history_key(key: str) -> str:
    """Normalize a W&B history key extracted from protobuf records.

    Some internal records expose keys with a trailing ``.``. We strip that
    suffix so the exported DataFrame matches the user-facing metric names.

    Args:
        key: Raw metric key from the W&B protobuf record.

    Returns:
        Normalized metric key.
    """
    return key[:-1] if key.endswith(".") else key


def _decode_history_value(value_json: str) -> Any:
    """Decode a history value stored as JSON text.

    Args:
        value_json: Raw JSON payload from a history item.

    Returns:
        Decoded Python object when possible, otherwise the original string.
    """
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return value_json


def load_history_dataframe(run_dir: Path) -> pd.DataFrame:
    """Load the full history table from an offline W&B directory.

    Args:
        run_dir: Path to an offline W&B run directory.

    Returns:
        A DataFrame whose rows correspond to W&B history records.

    Raises:
        FileNotFoundError: If no ``run-*.wandb`` file exists in ``run_dir``.
        ValueError: If the parsed history is empty.
    """
    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        raise FileNotFoundError(f"No run-*.wandb file found in {run_dir}.")

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))

    rows: list[dict[str, Any]] = []
    while True:
        payload = ds.scan_data()
        if payload is None:
            break

        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        if record.WhichOneof("record_type") != "history":
            continue

        row: dict[str, Any] = {}
        for item in record.history.item:
            prefix = list(item.nested_key)
            key_parts = prefix + [item.key]
            key = _normalize_history_key(".".join(key_parts) if prefix else item.key)
            if item.value_json:
                row[key] = _decode_history_value(item.value_json)

        if row:
            rows.append(row)

    if not rows:
        raise ValueError(f"No history records were found in {run_dir}.")

    history_df = pd.DataFrame(rows)
    if "_step" in history_df.columns:
        history_df = history_df.sort_values("_step", kind="stable")
    return history_df.reset_index(drop=True)


def load_summary(run_dir: Path) -> dict[str, Any]:
    """Load the W&B summary JSON if available.

    Args:
        run_dir: Path to an offline W&B run directory.

    Returns:
        Parsed summary dictionary, or an empty dictionary when absent.
    """
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(run_dir: Path) -> dict[str, Any]:
    """Load the W&B config YAML if available.

    Args:
        run_dir: Path to an offline W&B run directory.

    Returns:
        Parsed config dictionary, or an empty dictionary when absent.
    """
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_step_series(history_df: pd.DataFrame) -> pd.Series:
    """Return the x-axis series used for all plots.

    Args:
        history_df: Parsed W&B history table.

    Returns:
        Step series if present, otherwise a zero-based index series.
    """
    if "_step" in history_df.columns:
        return pd.to_numeric(history_df["_step"], errors="coerce")
    return pd.Series(range(len(history_df)), name="index")


def _sanitize_filename(name: str) -> str:
    """Convert an arbitrary metric group name into a filesystem-safe stem.

    Args:
        name: Raw metric group name.

    Returns:
        A sanitized filename stem.
    """
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "root"


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks.

    Args:
        items: Input list.
        chunk_size: Maximum number of items per chunk.

    Returns:
        Chunked list.
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _build_numeric_history(
    history_df: pd.DataFrame,
    min_non_null_fraction: float,
) -> pd.DataFrame:
    """Extract a numeric-only history table for visualization.

    Args:
        history_df: Raw parsed W&B history table.
        min_non_null_fraction: Minimum fraction of non-null values required.

    Returns:
        DataFrame containing only numeric metric columns suitable for plotting.
    """
    min_non_null_count = max(1, int(len(history_df) * min_non_null_fraction))
    numeric_columns: dict[str, pd.Series] = {}
    for column in history_df.columns:
        if column in RESERVED_X_COLUMNS:
            continue
        numeric_series = pd.to_numeric(history_df[column], errors="coerce")
        if numeric_series.notna().sum() < min_non_null_count:
            continue
        if numeric_series.dropna().nunique() <= 1:
            continue
        numeric_columns[column] = numeric_series
    return pd.DataFrame(numeric_columns, index=history_df.index)


def _group_metric_name(metric_name: str) -> str:
    """Map a metric name to a plotting group based on its namespace.

    Args:
        metric_name: Full metric name.

    Returns:
        Group name used to organize output figures.
    """
    for separator in ("/", "."):
        if separator in metric_name:
            return metric_name.split(separator)[0]
    return "ungrouped"


def _compute_metric_stats(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics used to rank and filter metrics.

    Args:
        numeric_df: Numeric-only metric table.

    Returns:
        DataFrame indexed by metric name with plotting statistics.
    """
    records: list[dict[str, Any]] = []
    for column in numeric_df.columns:
        finite = numeric_df[column].dropna()
        if finite.empty:
            continue
        records.append(
            {
                "metric": column,
                "group": _group_metric_name(column),
                "count": int(finite.shape[0]),
                "variance": float(finite.var()) if finite.shape[0] > 1 else 0.0,
                "range": float(finite.max() - finite.min()),
                "mean": float(finite.mean()),
                "std": float(finite.std()) if finite.shape[0] > 1 else 0.0,
            }
        )

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        return stats_df
    return stats_df.sort_values(
        by=["variance", "range", "count", "metric"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _select_overview_metrics(metric_stats_df: pd.DataFrame, top_k_overview: int) -> list[str]:
    """Select informative metrics for the overview figure.

    Args:
        metric_stats_df: Per-metric summary table.
        top_k_overview: Maximum number of overview metrics.

    Returns:
        Ordered metric name list for overview plotting.
    """
    if metric_stats_df.empty:
        return []

    selected: list[str] = []
    seen_groups: set[str] = set()

    for _, row in metric_stats_df.iterrows():
        metric = str(row["metric"])
        group = str(row["group"])
        if group not in seen_groups:
            selected.append(metric)
            seen_groups.add(group)
        if len(selected) >= top_k_overview:
            return selected

    for metric in metric_stats_df["metric"].tolist():
        if metric not in selected:
            selected.append(metric)
        if len(selected) >= top_k_overview:
            break
    return selected


def _plot_metric_group(
    history_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    metrics: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot one metric group with dynamic subplot layout.

    Args:
        history_df: Raw parsed W&B history table.
        numeric_df: Numeric-only metric table.
        metrics: Metrics included in this figure.
        title: Figure title.
        output_path: Destination PNG path.
    """
    if not metrics:
        return

    steps = _get_step_series(history_df)
    num_panels = len(metrics)
    num_cols = min(2, num_panels)
    num_rows = math.ceil(num_panels / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(7 * num_cols, 3.8 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics, strict=False):
        ax.plot(steps, numeric_df[metric], linewidth=1.8)
        ax.set_title(metric)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, linestyle="--")

    for ax in axes_flat[num_panels:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _extract_lambda_eff_metric_pairs(numeric_df: pd.DataFrame) -> dict[str, dict[str, dict[str, str]]]:
    """Collect lambda_eff h/w metrics grouped by sampled timestep and statistic.

    Expected metric names follow the pattern
    ``lambda_eff/t{timestep}/{axis}_{statistic}``, where ``axis`` is ``h`` or
    ``w`` and ``statistic`` is one of ``min``, ``max``, or ``mean``.

    Args:
        numeric_df: Numeric-only metric table.

    Returns:
        Nested mapping ``{timestep: {statistic: {axis: metric_name}}}``.
    """
    pattern = re.compile(r"^lambda_eff/t(?P<timestep>\d+)/(?:)(?P<axis>[hw])_(?P<stat>min|max|mean)$")
    grouped_metrics: dict[str, dict[str, dict[str, str]]] = {}
    for metric_name in numeric_df.columns:
        match = pattern.match(metric_name)
        if match is None:
            continue
        timestep = match.group("timestep")
        axis = match.group("axis")
        stat = match.group("stat")
        grouped_metrics.setdefault(timestep, {}).setdefault(stat, {})[axis] = metric_name
    return grouped_metrics


def _plot_lambda_eff_pdf(
    history_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    timestep: str,
    stat_name: str,
    metric_pair: dict[str, str],
    output_path: Path,
) -> None:
    """Plot one lambda_eff PDF figure with height/width overlaid.

    Args:
        history_df: Raw parsed W&B history table.
        numeric_df: Numeric-only metric table.
        timestep: Sampled diffusion timestep identifier.
        stat_name: Statistic name, such as ``min``, ``max``, or ``mean``.
        metric_pair: Mapping from axis name to concrete metric column.
        output_path: Destination PDF path.
    """
    height_metric = metric_pair.get("h")
    width_metric = metric_pair.get("w")
    if height_metric is None or width_metric is None:
        return

    steps = _get_step_series(history_df)
    height_series = pd.to_numeric(numeric_df[height_metric], errors="coerce")
    width_series = pd.to_numeric(numeric_df[width_metric], errors="coerce")
    visible_mask = steps.notna() & (steps >= 0) & (steps <= 1000)
    visible_steps = steps.loc[visible_mask]
    visible_height_series = height_series.loc[visible_mask]
    visible_width_series = width_series.loc[visible_mask]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(
        visible_steps,
        visible_height_series,
        linewidth=2.0,
        color=LAMBDA_EFF_HEIGHT_COLOR,
        label="height",
    )
    ax.plot(
        visible_steps,
        visible_width_series,
        linewidth=2.0,
        color=LAMBDA_EFF_WIDTH_COLOR,
        label="width",
    )
    ax.set_xlim(0, 1000)
    if stat_name == "mean":
        ax.set_ylim(0.90, 1.10)
        ax.set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        combined_series = pd.concat([visible_height_series, visible_width_series], axis=0).dropna()
        if not combined_series.empty:
            y_min = float(combined_series.min())
            y_max = float(combined_series.max())
            if y_min == y_max:
                epsilon = max(abs(y_min) * 1e-4, 1e-6)
                ax.set_ylim(y_min - epsilon, y_max + epsilon)
            else:
                ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Step", fontsize=LAMBDA_EFF_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Value", fontsize=LAMBDA_EFF_AXIS_LABEL_FONT_SIZE)
    ax.set_box_aspect(1)
    ax.tick_params(axis="both", labelsize=LAMBDA_EFF_TICK_FONT_SIZE)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=LAMBDA_EFF_LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def create_lambda_eff_pdf_figures(
    output_dir: Path,
    history_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
) -> None:
    """Create extra lambda_eff PDF figures when the required metrics exist.

    For each sampled timestep ``t{x}``, this function creates a subdirectory
    named ``lambda_eff_t{x}`` under ``figures`` and saves three PDF files:
    ``lambda_eff_hw_min.pdf``, ``lambda_eff_hw_max.pdf``, and
    ``lambda_eff_hw_mean.pdf``. Each figure overlays the height and width
    curves for that statistic.

    Args:
        output_dir: Destination directory for all visualizations.
        history_df: Raw parsed W&B history table.
        numeric_df: Numeric-only metric table.
    """
    lambda_eff_groups = _extract_lambda_eff_metric_pairs(numeric_df)
    if not lambda_eff_groups:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for timestep, stat_groups in sorted(lambda_eff_groups.items(), key=lambda item: int(item[0])):
        timestep_dir = figures_dir / f"lambda_eff_t{timestep}"
        timestep_dir.mkdir(parents=True, exist_ok=True)
        for stat_name in ("min", "max", "mean"):
            metric_pair = stat_groups.get(stat_name)
            if metric_pair is None:
                continue
            _plot_lambda_eff_pdf(
                history_df=history_df,
                numeric_df=numeric_df,
                timestep=timestep,
                stat_name=stat_name,
                metric_pair=metric_pair,
                output_path=timestep_dir / f"lambda_eff_hw_{stat_name}.pdf",
            )


def export_metadata(
    output_dir: Path,
    run_dir: Path,
    history_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    metric_stats_df: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Write parsed tabular data and metadata to disk.

    Args:
        output_dir: Destination directory for exported files.
        run_dir: Source offline W&B directory.
        history_df: Parsed W&B history table.
        numeric_df: Numeric-only metric table.
        metric_stats_df: Per-metric summary table.
        summary: Final W&B summary dictionary.
        config: W&B config dictionary.
    """
    history_df.to_csv(output_dir / "history.csv", index=False)
    numeric_df.to_csv(output_dir / "numeric_history.csv", index=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metric_stats_df.to_csv(output_dir / "metric_stats.csv", index=False)

    groups: dict[str, list[str]] = {}
    for metric in numeric_df.columns.tolist():
        group = _group_metric_name(metric)
        groups.setdefault(group, []).append(metric)

    metadata = {
        "run_dir": str(run_dir),
        "num_history_rows": int(len(history_df)),
        "num_history_columns": int(len(history_df.columns)),
        "num_numeric_metrics": int(len(numeric_df.columns)),
        "metric_names": sorted(history_df.columns.tolist()),
        "numeric_metric_names": sorted(numeric_df.columns.tolist()),
        "metric_groups": {group: sorted(metrics) for group, metrics in sorted(groups.items())},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def create_figures(
    output_dir: Path,
    history_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    metric_stats_df: pd.DataFrame,
    max_metrics_per_figure: int,
    top_k_overview: int,
) -> None:
    """Generate generic metric visualizations for arbitrary W&B runs.

    Args:
        output_dir: Destination directory for figures.
        history_df: Parsed W&B history table.
        numeric_df: Numeric-only metric table.
        metric_stats_df: Per-metric summary table.
        max_metrics_per_figure: Maximum number of metrics per figure.
        top_k_overview: Maximum number of metrics in the overview figure.
    """
    if numeric_df.empty or metric_stats_df.empty:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overview_metrics = _select_overview_metrics(metric_stats_df, top_k_overview)
    overview_chunks = _chunk_list(overview_metrics, max_metrics_per_figure)
    for chunk_index, metrics in enumerate(overview_chunks, start=1):
        suffix = f"_part{chunk_index:02d}" if len(overview_chunks) > 1 else ""
        _plot_metric_group(
            history_df=history_df,
            numeric_df=numeric_df,
            metrics=metrics,
            title="Overview: high-variance metrics",
            output_path=figures_dir / f"overview{suffix}.png",
        )

    for group_name, group_df in metric_stats_df.groupby("group", sort=True):
        metrics = group_df["metric"].tolist()
        metric_chunks = _chunk_list(metrics, max_metrics_per_figure)
        file_stem = _sanitize_filename(group_name)
        for chunk_index, chunk_metrics in enumerate(metric_chunks, start=1):
            suffix = f"_part{chunk_index:02d}" if len(metric_chunks) > 1 else ""
            _plot_metric_group(
                history_df=history_df,
                numeric_df=numeric_df,
                metrics=chunk_metrics,
                title=f"Metric group: {group_name}",
                output_path=figures_dir / f"group_{file_stem}{suffix}.png",
            )

    create_lambda_eff_pdf_figures(
        output_dir=output_dir,
        history_df=history_df,
        numeric_df=numeric_df,
    )


def main() -> None:
    """Run the offline W&B visualization pipeline."""
    args = parse_args()
    run_dir = args.wandb_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df = load_history_dataframe(run_dir)
    numeric_df = _build_numeric_history(
        history_df=history_df,
        min_non_null_fraction=args.min_non_null_fraction,
    )
    metric_stats_df = _compute_metric_stats(numeric_df)
    summary = load_summary(run_dir)
    config = load_config(run_dir)

    export_metadata(
        output_dir=output_dir,
        run_dir=run_dir,
        history_df=history_df,
        numeric_df=numeric_df,
        metric_stats_df=metric_stats_df,
        summary=summary,
        config=config,
    )
    create_figures(
        output_dir=output_dir,
        history_df=history_df,
        numeric_df=numeric_df,
        metric_stats_df=metric_stats_df,
        max_metrics_per_figure=args.max_metrics_per_figure,
        top_k_overview=args.top_k_overview,
    )

    print(f"Loaded {len(history_df)} history rows from: {run_dir}")
    print(f"Discovered {len(numeric_df.columns)} numeric metrics for visualization")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
