"""Kıyas çıtaları. LSTM'in "iyi" sayılması için bunları geçmesi gerekir.

1. Son değer:          OT(t+h) = OT(t)
2. Dünün aynı saati:   OT(t+h) = OT(t+h-24)
3. Doğrusal regresyon: elle seçilmiş gecikme özellikleri -> değişim
4. Gradient boosting:  aynı özellikler, %5/%50/%95 quantile -> aralık tahmini

Çalıştırma: .venv\\Scripts\\python.exe src\\baselines.py
"""
import time

import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from data import FEATURE_COLS, HORIZONS, QUANTILES, TARGET, TIME_COLS, WINDOW, prepare, to_celsius
from metrics import interval_metrics, point_metrics

OT_IDX = FEATURE_COLS.index(TARGET)
N_LOAD = len(FEATURE_COLS) - 1
OT_LAGS = list(range(24)) + [47, 71, 167]   # son 24 saat + 2, 3 ve 7 gün önce


def tabular(X: np.ndarray, relative: bool = True) -> np.ndarray:
    """Pencereyi düz özellik vektörüne çevirir (ağaç ve doğrusal modeller dizi göremez).

    relative=True: OT gecikmeleri şu anki OT'ye göre fark olarak verilir, mutlak seviye
    hiç verilmez. Mutlak seviyeyle eğitilen ağaç "30 °C'de şöyle değişir" gibi kurallar
    öğreniyor ve kış testinde (train'de az görülen seviyeler) çöküyordu.
    """
    last = WINDOW - 1
    ot_lags = X[:, [last - l for l in OT_LAGS], OT_IDX]
    if relative:
        ot_lags = ot_lags[:, 1:] - ot_lags[:, :1]
    loads_now = X[:, last, :N_LOAD]
    loads_mean24 = X[:, -24:, :N_LOAD].mean(axis=1)
    time_now = X[:, last, len(FEATURE_COLS):]
    return np.hstack([ot_lags, loads_now, loads_mean24, time_now])


def naive_preds(X, base, scaler):
    """Model gerektirmeyen tahminler, °C."""
    mean, std = float(scaler.mean[TARGET]), scaler.ot_std
    ot_c = X[:, :, OT_IDX] * std + mean            # pencere içindeki OT, °C
    last = np.repeat(base[:, None], len(HORIZONS), axis=1)
    yesterday = np.stack([ot_c[:, WINDOW - 1 + h - 24] for h in HORIZONS], axis=1)
    return {"son_deger": last, "dunun_ayni_saati": yesterday}


def main():
    data, scaler = prepare("delta")
    Xtr, ytr, _, _ = data["train"]
    results = {}

    for split_name in ["val", "test"]:
        X, y, base, _ = data[split_name]
        y_c = to_celsius(y, base, scaler, "delta")
        for name, pred in naive_preds(X, base, scaler).items():
            results.setdefault(name, {}).update(point_metrics(y_c, pred, split_name))

    params = {"son_deger": {}, "dunun_ayni_saati": {}}
    for relative in [False, True]:
        suffix = "goreli" if relative else "mutlak"
        Ftr = tabular(Xtr, relative)

        # Doğrusal regresyon (Ridge: küçük L2 cezası, gecikmeler birbirine çok benzediği için kararlı)
        name = f"ridge_{suffix}"
        params[name] = {"features": suffix}
        t0 = time.time()
        lin = [Ridge(alpha=1.0).fit(Ftr, ytr[:, i]) for i in range(len(HORIZONS))]
        results[name] = {"train_seconds": time.time() - t0}
        for split_name in ["val", "test"]:
            X, y, base, _ = data[split_name]
            F = tabular(X, relative)
            pred = np.stack([m.predict(F) for m in lin], axis=1)
            results[name].update(point_metrics(
                to_celsius(y, base, scaler, "delta"), to_celsius(pred, base, scaler, "delta"), split_name))

        # Gradient boosting, her ufuk ve her yüzdelik için ayrı model
        name = f"gbr_quantile_{suffix}"
        params[name] = {"features": suffix, "max_iter": 300, "learning_rate": 0.05}
        t0 = time.time()
        gbr = {(i, q): HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                                     learning_rate=0.05, random_state=0).fit(Ftr, ytr[:, i])
               for i in range(len(HORIZONS)) for q in QUANTILES}
        results[name] = {"train_seconds": time.time() - t0}
        for split_name in ["val", "test"]:
            X, y, base, _ = data[split_name]
            F = tabular(X, relative)
            pq = np.stack([np.stack([gbr[(i, q)].predict(F) for q in QUANTILES], axis=1)
                           for i in range(len(HORIZONS))], axis=1)        # (n, H, Q)
            pq_c = to_celsius(pq, base, scaler, "delta")
            y_c = to_celsius(y, base, scaler, "delta")
            results[name].update(point_metrics(y_c, pq_c[:, :, 1], split_name))
            results[name].update(interval_metrics(y_c, pq_c[:, :, 0], pq_c[:, :, 2], split_name))

    for name, metrics in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", "baseline")
            mlflow.log_params({"target": "delta", **params[name]})
            mlflow.log_metrics(metrics)

    print(f"{'model':18s}" + "".join(f"  val_h{h:<3d} test_h{h:<3d}" for h in HORIZONS) + "  (MAE °C)")
    for name, m in results.items():
        print(f"{name:18s}" + "".join(f"  {m[f'val_mae_h{h}']:7.3f} {m[f'test_mae_h{h}']:7.3f}" for h in HORIZONS))
    for name in ["gbr_quantile_mutlak", "gbr_quantile_goreli"]:
        g = results[name]
        print(f"\n{name} %90 aralık kapsaması: " + "  ".join(
            f"h{h}: val={g[f'val_coverage_h{h}']:.2f} test={g[f'test_coverage_h{h}']:.2f}" for h in HORIZONS))


if __name__ == "__main__":
    main()
