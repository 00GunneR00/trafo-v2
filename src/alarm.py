"""Alarm kuralları. Hem değerlendirme (anomaly.py) hem API kullanır."""
import numpy as np

from data import HORIZONS, TARGET
from forecaster import Forecaster


def signals(fc: Forecaster, df, s_start: int, s_end: int) -> dict:
    """[s_start, s_end) saatleri için, her ufukta (s - h) anındaki tahmine göre:
    exceed: (T, H) bool, alarm sınırı aşıldı mı?
    z:      (T, H) ölçekli artık = (gerçek - ortanca) / ölçek
    median, lower, upper: (T, H) °C, (s - h) anında s için yapılmış tahmin ve kalibre %90 aralık
    """
    max_h = max(HORIZONS)
    t = np.arange(s_start - max_h, s_end)
    pred = fc.predict(df, t)                                   # (len(t), H, Q)
    alarm_upper = fc.upper_bound(pred, "alarm")
    upper = fc.upper_bound(pred, "interval")
    lower = fc.lower_bound(pred)
    # Ölçek: %90 aralığın yarı genişliği / 1,645 (normal dağılımda bir standart sapma)
    scale = (upper - lower) / 2 / 1.645
    ot = df[TARGET].to_numpy(np.float32)
    s = np.arange(s_start, s_end)
    out = {k: [] for k in ["exceed", "z", "median", "lower", "upper"]}
    for i, h in enumerate(HORIZONS):
        row = s - h - (s_start - max_h)
        out["exceed"].append(ot[s] > alarm_upper[row, i])
        out["z"].append((ot[s] - pred[row, i, 1]) / scale[row, i])
        out["median"].append(pred[row, i, 1])
        out["lower"].append(lower[row, i])
        out["upper"].append(upper[row, i])
    return {k: np.stack(v, axis=1) for k, v in out.items()}


def alarms_threshold(exceed: np.ndarray, n: int) -> np.ndarray:
    """Art arda n saat aşım olan saatler True. exceed: (T,) bool."""
    run = np.zeros(len(exceed), dtype=int)
    count = 0
    for i, e in enumerate(exceed):
        count = count + 1 if e else 0
        run[i] = count
    return run >= n


def cusum_path(z: np.ndarray, k: float, h_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Tek yönlü (yukarı) CUSUM: S = max(0, S + z - k); S > H ise alarm.

    k: "normal" sayılan sapma payı; z bunun altındaysa S azalır.
    H: alarm eşiği. Alarm verince S sıfırlanır, böylece her olay ayrı sayılır.
    Döner: (her saatin S değeri, alarm bool)
    """
    path = np.zeros(len(z), dtype=np.float32)
    out = np.zeros(len(z), dtype=bool)
    s = 0.0
    for i, v in enumerate(z):
        s = max(0.0, s + float(v) - k)
        path[i] = s
        if s > h_thr:
            out[i] = True
            s = 0.0
    return path, out


def alarms_cusum(z: np.ndarray, k: float, h_thr: float) -> np.ndarray:
    return cusum_path(z, k, h_thr)[1]


def alarm_fn(rule: dict):
    if rule["kind"] == "esik":
        return lambda sig, i: alarms_threshold(sig["exceed"][:, i], rule["n"])
    return lambda sig, i: alarms_cusum(sig["z"][:, i], rule["k"], rule["H"])
