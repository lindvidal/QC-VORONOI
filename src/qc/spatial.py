import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..config import P1_VALUES, P2_VALUES, QUALITY_THRESHOLD


@dataclass
class QCResult:
    gauge_code: str
    year: int
    mit_minutes: int
    p1: float
    p2: float
    predicted_quality: str
    true_quality: Optional[str]
    yearly_rainfall_target: float
    yearly_rainfall_neighbors: float
    n_neighbors: int


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        denom = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def find_delaunay_neighbors(
    target_point: Point,
    reference_points: pd.DataFrame,
    reference_coords: np.ndarray,
    tri: Delaunay,
) -> List[str]:
    simplex_indices = tri.find_simplex(np.array([[target_point.x, target_point.y]]))
    if simplex_indices[0] < 0:
        return []
    simplex = tri.simplices[simplex_indices[0]]
    neighbor_codes = reference_points.iloc[simplex]["gauge_code"].tolist()
    return neighbor_codes


def compute_neighbor_statistics(
    df_events: pd.DataFrame,
    neighbor_codes: List[str],
    gauge_info: pd.DataFrame,
) -> Dict[str, float]:
    neighbor_data = df_events[df_events["gauge_code"].isin(neighbor_codes)]
    if neighbor_data.empty:
        return {}
    return {
        "yearly_rainfall_mean": neighbor_data["yearly_rainfall"].mean(),
        "rainfall_event_mean": neighbor_data["rainfall_events"].mean(),
        "rainfall_intensity_mean": neighbor_data["mean_rainfall_intensity"].mean(),
        "rainfall_depth_mean": neighbor_data["mean_rainfall_depth"].mean(),
        "rainfall_duration_mean": neighbor_data["mean_rainfall_duration"].mean(),
        "dry_time_mean": neighbor_data["mean_dry_time"].mean(),
    }


def evaluate_quality_factor(target_value: float, neighbor_mean: float, p1: float, p2: float) -> str:
    if neighbor_mean * p2 <= target_value <= neighbor_mean * p1:
        return "HQ"
    return "LQ"


def evaluate_hybrid_quality(quality_results: Dict[str, str]) -> str:
    hq_count = sum(1 for v in quality_results.values() if v == "HQ")
    if hq_count >= QUALITY_THRESHOLD:
        return "HQ"
    return "LQ"


def run_qc_delaunay(
    target_row: pd.Series,
    reference_df: pd.DataFrame,
    target_year: int,
    mit_minutes: int,
    p1: float,
    p2: float,
    gauge_info: pd.DataFrame,
) -> Dict[str, str]:
    lt = gauge_info[gauge_info["gauge_code"] == target_row["gauge_code"]]
    if lt.empty:
        return {}
    target_code = lt.iloc[0]["gauge_code"]
    lat, lon = lt.iloc[0]["lat"], lt.iloc[0]["long"]
    target_point = Point(lon, lat)
    ref_codes = reference_df["gauge_code"].unique()
    ref_points = gauge_info[gauge_info["gauge_code"].isin(ref_codes) & (gauge_info["gauge_code"] != target_code)].copy()
    ref_points = ref_points.dropna(subset=["lat", "long"])
    if len(ref_points) < 3:
        return {}
    coords = ref_points[["long", "lat"]].values
    tri = Delaunay(coords)
    neighbors = find_delaunay_neighbors(target_point, ref_points, coords, tri)
    if not neighbors:
        return {}
    neighbor_stats = compute_neighbor_statistics(reference_df, neighbors, gauge_info)
    if not neighbor_stats:
        return {}
    quality = {
        "yearly_rainfall_quality": evaluate_quality_factor(
            target_row["yearly_rainfall"], neighbor_stats["yearly_rainfall_mean"], p1, p2
        ),
        "rainfall_event_quality": evaluate_quality_factor(
            target_row["rainfall_events"], neighbor_stats["rainfall_event_mean"], p1, p2
        ),
        "rainfall_intensity_quality": evaluate_quality_factor(
            target_row["mean_rainfall_intensity"], neighbor_stats["rainfall_intensity_mean"], p1, p2
        ),
        "rainfall_duration_quality": evaluate_quality_factor(
            target_row["mean_rainfall_duration"], neighbor_stats["rainfall_duration_mean"], p1, p2
        ),
    }
    quality["hybrid_quality"] = evaluate_hybrid_quality(quality)
    return quality


def compute_confusion_matrix(predictions: List[str], true_labels: List[str], positive_class: str = "HQ") -> ConfusionMatrix:
    cm = ConfusionMatrix()
    for pred, true in zip(predictions, true_labels):
        if pred == positive_class and true == positive_class:
            cm.tp += 1
        elif pred == positive_class and true != positive_class:
            cm.fp += 1
        elif pred != positive_class and true != positive_class:
            cm.tn += 1
        else:
            cm.fn += 1
    return cm


def fine_tune_p1_p2(
    df_events: pd.DataFrame,
    df_quality: pd.DataFrame,
    gauge_info: pd.DataFrame,
    mit_minutes: int,
    target_year: int,
    reference_years: List[int],
) -> pd.DataFrame:
    ref_df = df_events[df_events["year"].isin(reference_years)]
    tgt_df = df_events[df_events["year"] == target_year]
    quality_lookup = df_quality.set_index("gauge_code")["quality"].to_dict() if not df_quality.empty else {}

    results = []
    for p1 in P1_VALUES:
        for p2 in P2_VALUES:
            predictions = []
            true_labels = []
            for _, tgt_row in tgt_df.iterrows():
                qc = run_qc_delaunay(tgt_row, ref_df, target_year, mit_minutes, p1, p2, gauge_info)
                if qc:
                    pred = qc["hybrid_quality"]
                    code = tgt_row["gauge_code"]
                    true_label = quality_lookup.get(code, "LQ")
                    predictions.append(pred)
                    true_labels.append(true_label)
            if predictions:
                cm = compute_confusion_matrix(predictions, true_labels)
                results.append({
                    "p1": p1, "p2": p2,
                    "accuracy": cm.accuracy, "precision": cm.precision,
                    "recall": cm.recall, "f1": cm.f1,
                    "tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn,
                })
    return pd.DataFrame(results)
