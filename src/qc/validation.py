"""
Statistical validation methods for QC results:
- Bootstrap confidence intervals
- McNemar test for model comparison
- Temporal cross-validation (k-fold by year)
- Permutation test (model vs random)
- Sensitivity surface analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from typing import Callable, List, Optional, Dict

from ..config import P1_VALUES, P2_VALUES, QUALITY_FACTORS
from .spatial import run_qc_delaunay, compute_confusion_matrix, fine_tune_p1_p2


# ─── 1. Bootstrap Confidence Intervals ───────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = 5000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap confidence interval for any classification metric.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    metric_fn : callable
        Function that takes (y_true, y_pred) and returns a float score.
    n_iter : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    dict with 'lower', 'upper', 'mean', 'std'.
    """
    rng = np.random.default_rng()
    n = len(y_true)
    scores = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "ci": ci,
        "n_iter": n_iter,
    }


def _f1_wrapper(y_true, y_pred):
    cm = compute_confusion_matrix(y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                                   y_true.tolist() if hasattr(y_true, 'tolist') else y_true)
    return cm.f1


def _precision_wrapper(y_true, y_pred):
    cm = compute_confusion_matrix(y_pred, y_true)
    return cm.precision


def _recall_wrapper(y_true, y_pred):
    cm = compute_confusion_matrix(y_pred, y_true)
    return cm.recall


def _accuracy_wrapper(y_true, y_pred):
    cm = compute_confusion_matrix(y_pred, y_true)
    return cm.accuracy


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iter: int = 5000,
    ci: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CI for F1, precision, recall, accuracy simultaneously."""
    wrappers = {
        "f1": _f1_wrapper,
        "precision": _precision_wrapper,
        "recall": _recall_wrapper,
        "accuracy": _accuracy_wrapper,
    }
    results = {}
    for name, fn in wrappers.items():
        results[name] = bootstrap_ci(y_true, y_pred, fn, n_iter=n_iter, ci=ci)
    return results


# ─── 2. McNemar Test ────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict:
    """McNemar's test to compare two classification models.

    Counts:
        b: model A correct, model B wrong
        c: model A wrong, model B correct

    H0: both models have equal error rate.
    """
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)

    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))

    numerator = (abs(b - c) - 1) ** 2
    denominator = b + c
    if denominator == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = numerator / denominator
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "b_modelA_correct_B_wrong": b,
        "c_modelA_wrong_B_correct": c,
        "n_discordant": b + c,
        "significant": p_value < 0.05,
    }


# ─── 3. Temporal Cross-Validation ───────────────────────────────────

def temporal_cross_validation(
    df_events: pd.DataFrame,
    df_quality: pd.DataFrame,
    gauge_info: pd.DataFrame,
    mit_minutes: int,
    years: Optional[List[int]] = None,
    p1: float = 1.6,
    p2: float = 0.8,
) -> pd.DataFrame:
    """Temporal k-fold CV: each year is test, all others are reference.

    Returns a DataFrame with per-fold and aggregate metrics.
    """
    all_years = sorted(df_events['year'].unique()) if years is None else sorted(years)
    if len(all_years) < 2:
        raise ValueError("Need at least 2 years for temporal CV")

    quality_lookup = df_quality.set_index("gauge_code")["quality"].to_dict() if not df_quality.empty else {}

    records = []
    for test_year in all_years:
        ref_years = [y for y in all_years if y != test_year]
        predictions = []
        true_labels = []

        ref_df = df_events[df_events['year'].isin(ref_years)]
        tgt_df = df_events[df_events['year'] == test_year]

        for _, tgt_row in tgt_df.iterrows():
            qc = run_qc_delaunay(tgt_row, ref_df, test_year, mit_minutes, p1, p2, gauge_info)
            if qc:
                pred = qc['hybrid_quality']
                code = tgt_row['gauge_code']
                true_label = quality_lookup.get(code, "LQ")
                predictions.append(pred)
                true_labels.append(true_label)

        if predictions:
            cm = compute_confusion_matrix(predictions, true_labels)
            records.append({
                'test_year': test_year,
                'n_ref_years': len(ref_years),
                'n_test': len(predictions),
                'accuracy': cm.accuracy,
                'precision': cm.precision,
                'recall': cm.recall,
                'f1': cm.f1,
                'tp': cm.tp, 'fp': cm.fp, 'tn': cm.tn, 'fn': cm.fn,
            })

    df_cv = pd.DataFrame(records)

    if not df_cv.empty:
        agg = {
            'test_year': 'overall_mean',
            'n_ref_years': int(df_cv['n_ref_years'].mean()),
            'n_test': int(df_cv['n_test'].mean()),
            'accuracy': df_cv['accuracy'].mean(),
            'precision': df_cv['precision'].mean(),
            'recall': df_cv['recall'].mean(),
            'f1': df_cv['f1'].mean(),
            'tp': int(df_cv['tp'].mean()),
            'fp': int(df_cv['fp'].mean()),
            'tn': int(df_cv['tn'].mean()),
            'fn': int(df_cv['fn'].mean()),
        }
        agg['f1_std'] = df_cv['f1'].std()
        agg['accuracy_std'] = df_cv['accuracy'].std()
        agg['precision_std'] = df_cv['precision'].std()
        agg['recall_std'] = df_cv['recall'].std()

        df_agg = pd.DataFrame([agg])
        return pd.concat([df_cv, df_agg], ignore_index=True)

    return df_cv


def temporal_cv_fine_tune(
    df_events: pd.DataFrame,
    df_quality: pd.DataFrame,
    gauge_info: pd.DataFrame,
    mit_minutes: int,
    years: List[int],
) -> pd.DataFrame:
    """Temporal CV that also fine-tunes p1/p2 on ref years for each fold.

    For each test year, fine-tune p1/p2 on ref years, then evaluate on test year.
    Returns per-fold best params and metrics.
    """
    quality_lookup = df_quality.set_index("gauge_code")["quality"].to_dict() if not df_quality.empty else {}
    all_years = sorted(years)
    records = []
    for test_year in all_years:
        ref_years = [y for y in all_years if y != test_year]
        ref_df = df_events[df_events['year'].isin(ref_years)]

        # Fine-tune on reference years (uses all ref years as reference)
        tuning = fine_tune_p1_p2(
            df_events=df_events[df_events['year'].isin(ref_years)],
            df_quality=df_quality,
            gauge_info=gauge_info,
            mit_minutes=mit_minutes,
            target_year=ref_years[-1],  # last ref year as pseudo-target for tuning
            reference_years=ref_years[:-1] if len(ref_years) > 1 else ref_years,
        )
        if tuning.empty:
            continue

        best = tuning.loc[tuning['f1'].idxmax()]
        best_p1, best_p2 = best['p1'], best['p2']

        # Evaluate on test year
        tgt_df = df_events[df_events['year'] == test_year]
        predictions = []
        true_labels = []
        for _, tgt_row in tgt_df.iterrows():
            qc = run_qc_delaunay(tgt_row, ref_df, test_year, mit_minutes, best_p1, best_p2, gauge_info)
            if qc:
                pred = qc['hybrid_quality']
                code = tgt_row['gauge_code']
                true_label = quality_lookup.get(code, "LQ")
                predictions.append(pred)
                true_labels.append(true_label)

        if predictions:
            cm = compute_confusion_matrix(predictions, true_labels)
            records.append({
                'test_year': test_year,
                'best_p1': best_p1,
                'best_p2': best_p2,
                'accuracy': cm.accuracy,
                'precision': cm.precision,
                'recall': cm.recall,
                'f1': cm.f1,
                'tp': cm.tp, 'fp': cm.fp, 'tn': cm.tn, 'fn': cm.fn,
            })

    df_result = pd.DataFrame(records)
    if not df_result.empty:
        agg = {
            'test_year': 'overall_mean',
            'best_p1': df_result['best_p1'].mean(),
            'best_p2': df_result['best_p2'].mean(),
            'accuracy': df_result['accuracy'].mean(),
            'precision': df_result['precision'].mean(),
            'recall': df_result['recall'].mean(),
            'f1': df_result['f1'].mean(),
        }
        agg['f1_std'] = df_result['f1'].std()
        df_result = pd.concat([df_result, pd.DataFrame([agg])], ignore_index=True)
    return df_result


# ─── 4. Permutation Test ────────────────────────────────────────────

def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable = _f1_wrapper,
    n_perm: int = 5000,
) -> Dict:
    """Test if model's metric is significantly better than random.

    Shuffles true labels N times, recomputes metric each time.
    p-value = proportion of permutations where shuffled metric >= real metric.
    """
    rng = np.random.default_rng()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    real_score = metric_fn(y_true, y_pred)

    perm_scores = np.zeros(n_perm)
    for i in range(n_perm):
        y_shuffled = rng.permutation(y_true)
        perm_scores[i] = metric_fn(y_shuffled, y_pred)

    p_value = (np.sum(perm_scores >= real_score) + 1) / (n_perm + 1)
    return {
        "real_score": round(real_score, 4),
        "perm_mean": round(float(np.mean(perm_scores)), 4),
        "perm_std": round(float(np.std(perm_scores)), 4),
        "p_value": round(p_value, 4),
        "n_perm": n_perm,
        "significant": p_value < 0.05,
    }


# ─── 5. Sensitivity Surface ─────────────────────────────────────────

def sensitivity_surface(
    df_events: pd.DataFrame,
    df_quality: pd.DataFrame,
    gauge_info: pd.DataFrame,
    mit_minutes: int,
    target_year: int,
    reference_years: List[int],
    p1_values: Optional[List[float]] = None,
    p2_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Full grid search over p1 x p2 with detailed per-combination metrics.

    Returns the same as fine_tune_p1_p2 but with additional columns
    for sensitivity analysis (youden_index, mcc, etc.).
    """
    if p1_values is None:
        p1_values = P1_VALUES
    if p2_values is None:
        p2_values = P2_VALUES

    df_grid = fine_tune_p1_p2(df_events, df_quality, gauge_info,
                               mit_minutes, target_year, reference_years)

    if df_grid.empty:
        return df_grid

    # Add derived metrics
    df_grid['youden_index'] = df_grid['recall'] + df_grid['precision'] - 1
    total = df_grid['tp'] + df_grid['fp'] + df_grid['tn'] + df_grid['fn']
    denom_mcc = np.sqrt((df_grid['tp'] + df_grid['fp']) * (df_grid['tp'] + df_grid['fn'])
                        * (df_grid['tn'] + df_grid['fp']) * (df_grid['tn'] + df_grid['fn']))
    df_grid['mcc'] = np.where(
        denom_mcc > 0,
        (df_grid['tp'] * df_grid['tn'] - df_grid['fp'] * df_grid['fn']) / denom_mcc,
        0.0,
    )
    df_grid['balanced_accuracy'] = (df_grid['recall'] + df_grid['tn'] / (df_grid['tn'] + df_grid['fp']).replace(0, np.nan)) / 2
    df_grid['f1_std_approx'] = df_grid['f1'] * np.sqrt(
        (1 - df_grid['precision']) / (df_grid['precision'] * total)
        + (1 - df_grid['recall']) / (df_grid['recall'] * total)
    )
    return df_grid


def find_plateau_region(
    df_surface: pd.DataFrame,
    metric: str = 'f1',
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Find the (p1,p2) region where metric is within tolerance of the max.

    Useful for identifying flat optima (multiple equally-good parameter pairs).
    """
    if df_surface.empty:
        return df_surface
    best_val = df_surface[metric].max()
    threshold = best_val - tolerance
    plateau = df_surface[df_surface[metric] >= threshold].copy()
    plateau['delta_to_best'] = best_val - plateau[metric]
    return plateau.sort_values(metric, ascending=False)
