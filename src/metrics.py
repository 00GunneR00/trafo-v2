"""Bütün modeller için ortak ölçüm ve MLflow kaydı."""
import numpy as np
import mlflow

from data import HORIZONS, ROOT

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("trafo-ot-tahmin")


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """y: (n, H) °C. Her ufuk için MAE ve RMSE."""
    out = {}
    for i, h in enumerate(HORIZONS):
        err = y_pred[:, i] - y_true[:, i]
        out[f"{prefix}_mae_h{h}"] = float(np.abs(err).mean())
        out[f"{prefix}_rmse_h{h}"] = float(np.sqrt((err ** 2).mean()))
    return out


def interval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, prefix: str) -> dict:
    """Kapsama: gerçek değerin aralığa düşme oranı (hedef 0,90). Genişlik: ortalama °C."""
    out = {}
    for i, h in enumerate(HORIZONS):
        inside = (y_true[:, i] >= lo[:, i]) & (y_true[:, i] <= hi[:, i])
        out[f"{prefix}_coverage_h{h}"] = float(inside.mean())
        out[f"{prefix}_width_h{h}"] = float((hi[:, i] - lo[:, i]).mean())
    return out


def true_celsius(y_scaled: np.ndarray, base: np.ndarray, scaler, target: str) -> np.ndarray:
    from data import to_celsius
    return to_celsius(y_scaled, base, scaler, target)
