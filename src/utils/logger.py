import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Optional

from ..config import ROOT


LOG_FILE = ROOT / "src" / "pipeline.log"


def _now() -> datetime:
    return datetime.now().astimezone()


def _ts() -> str:
    return _now().strftime("%Y-%m-%d %H:%M:%S%z")


def _ts_file() -> str:
    return _now().strftime("%Y%m%d_%H%M%S%z")


def _ensure_log():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text(
            "=== QC-VORONOI Pipeline Log ===\n"
            f"Criado em: {_ts()}\n"
            f"{'='*60}\n"
        )


def _append_log(entry: str):
    _ensure_log()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def _archive_path(filepath: Path) -> Path:
    archive_dir = filepath.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem
    suffix = filepath.suffix
    return archive_dir / f"{stem}_{_ts_file()}{suffix}"


def save_result(
    df_or_path: Union["pd.DataFrame", Path, str],
    name: str,
    subdir: str = "results",
    sep: str = ";",
    decimal: str = ",",
    index: bool = False,
    archive: bool = True,
) -> Path:
    """Save a DataFrame with timestamped archive copy + log entry.

    Parameters
    ----------
    df_or_path : DataFrame | Path | str
        Data to save, or path to an existing file to archive.
    name : str
        Filename (e.g. 'rainfall_events_sample.csv').
    subdir : str
        Subdirectory under src/ ('results' or 'figures').
    sep, decimal, index : passed to to_csv when df_or_path is a DataFrame.
    archive : bool
        Whether to create a timestamped archive copy.
    """
    name = Path(name).name
    import pandas as pd

    target_dir = ROOT / "src" / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / name

    if isinstance(df_or_path, pd.DataFrame):
        df_or_path.to_csv(filepath, sep=sep, decimal=decimal, index=index)
        source_desc = f"DataFrame ({len(df_or_path)} rows)"
    elif isinstance(df_or_path, (str, Path)):
        src = Path(df_or_path)
        if src.exists():
            shutil.copy2(src, filepath)
            source_desc = f"Copied from {src.name}"
        else:
            raise FileNotFoundError(f"Source not found: {src}")
    else:
        raise TypeError(f"Unsupported type: {type(df_or_path)}")

    entry = (
        f"[{_ts()}] SAVED  "
        f"{subdir}/{name}  |  {source_desc}"
    )
    print(entry)

    if archive:
        archived = _archive_path(filepath)
        shutil.copy2(filepath, archived)
        entry += f"\n{'':>27}ARCHIVE {archived.relative_to(ROOT)}"
        print(f"{'':>27}ARCHIVE {archived.relative_to(ROOT)}")

    _append_log(entry)
    return filepath


def save_figure(
    fig: "plt.Figure",
    name: str,
    subdir: str = "figures",
    dpi: Optional[int] = None,
    archive: bool = True,
    **kwargs,
) -> Path:
    """Save a matplotlib figure with timestamped archive copy + log entry."""
    name = Path(name).name
    from ..config import FIGURE_DPI

    dpi = dpi or FIGURE_DPI
    target_dir = ROOT / "src" / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / name

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", **kwargs)

    entry = (
        f"[{_ts()}] SAVED  "
        f"{subdir}/{name}  |  Figure"
    )
    print(entry)

    if archive:
        archived = _archive_path(filepath)
        fig.savefig(archived, dpi=dpi, bbox_inches="tight", **kwargs)
        entry += f"\n{'':>27}ARCHIVE {archived.relative_to(ROOT)}"
        print(f"{'':>27}ARCHIVE {archived.relative_to(ROOT)}")

    _append_log(entry)
    return filepath


def read_log(tail: Optional[int] = None) -> str:
    """Return the full log, or the last N lines."""
    _ensure_log()
    content = LOG_FILE.read_text(encoding="utf-8")
    if tail is not None:
        lines = content.strip().split("\n")
        return "\n".join(lines[-tail:])
    return content


def clear_log():
    """Truncate the log file (keeps header)."""
    _ensure_log()
    header = (
        "=== QC-VORONOI Pipeline Log ===\n"
        f"Reset em: {_ts()}\n"
        f"{'='*60}\n"
    )
    LOG_FILE.write_text(header)
