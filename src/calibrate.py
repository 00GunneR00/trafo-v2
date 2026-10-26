"""Aralıkları doğrulama verisiyle kalibre eder (conformalized quantile regression, CQR).

Ham model aralıkları %90 hedefine karşı val'de %93-94 kapsıyor (fazla geniş). CQR, val'de
"gerçek değer aralığın ne kadar dışında kaldı" skorlarına bakıp aralığı sabit bir miktar
genişletir ya da daraltır, böylece val kapsaması hedefe oturur.

  interval: iki yönlü, %90 aralık   skor = max(alt - y, y - üst)
  alarm:    tek yönlü, %99 üst sınır  skor = y - üst   (alarm için yanlış alarmı az tutmak istiyoruz)

Not: CQR'nin teorik garantisi örneklerin değiştirilebilir (exchangeable) olmasını ister; zaman
serisinde bu tam sağlanmaz. Bu yüzden kapsama test'te ayrıca ölçülür ve raporlanır.

Çalıştırma: python src\\calibrate.py   (sonuç models/latest.json içine yazılır)
"""
import json

import numpy as np

from data import HORIZONS, TARGET, TEST_END, TRAIN_END, VAL_END, load_raw
from forecaster import MODELS, Forecaster

INTERVAL_COVERAGE = 0.90
ALARM_COVERAGE = 0.99


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def evaluate(fc: Forecaster, df, start: int, end: int) -> dict:
    """[start, end) aralığındaki hedef saatler için kapsama ve genişlik."""
    ot = df[TARGET].to_numpy(np.float32)
    out = {}
    for i, h in enumerate(HORIZONS):
        t = np.arange(start - h, end - h)
        pred = fc.predict(df, t)
        y = ot[t + h]
        lo, hi = fc.lower_bound(pred)[:, i], fc.upper_bound(pred, "interval")[:, i]
        alarm_hi = fc.upper_bound(pred, "alarm")[:, i]
        out[f"h{h}"] = {
            "coverage_90": float(((y >= lo) & (y <= hi)).mean()),
            "width_90": float((hi - lo).mean()),
            "below_alarm_bound": float((y <= alarm_hi).mean()),
        }
    return out


def main():
    df = load_raw()
    fc = Forecaster()
    fc.calibration = {}
    before = {"val": evaluate(fc, df, TRAIN_END, VAL_END), "test": evaluate(fc, df, VAL_END, TEST_END)}

    ot = df[TARGET].to_numpy(np.float32)
    calib = {"interval": {}, "alarm": {}}
    for i, h in enumerate(HORIZONS):
        t = np.arange(TRAIN_END - h, VAL_END - h)
        pred = fc.predict(df, t)
        y = ot[t + h]
        lo, hi = pred[:, i, 0], pred[:, i, 2]
        calib["interval"][f"h{h}"] = conformal_quantile(np.maximum(lo - y, y - hi), INTERVAL_COVERAGE)
        calib["alarm"][f"h{h}"] = conformal_quantile(y - hi, ALARM_COVERAGE)

    fc.calibration = calib
    after = {"val": evaluate(fc, df, TRAIN_END, VAL_END), "test": evaluate(fc, df, VAL_END, TEST_END)}

    for path in [MODELS / "latest.json", MODELS / f"lstm_{fc.version}.json"]:
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["calibration"] = calib
        meta["calibration_report"] = {"before": before, "after": after}
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Kalibrasyon kayması (°C):", json.dumps(calib, indent=1))
    print(f"\n{'':6s}{'':10s}{'kapsama%90':>12s}{'genişlik':>10s}{'alarm sınırı altı':>20s}")
    for name, rep in [("önce", before), ("sonra", after)]:
        for split_name in ["val", "test"]:
            for h in HORIZONS:
                r = rep[split_name][f"h{h}"]
                print(f"{name:6s}{split_name + ' h' + str(h):10s}{r['coverage_90']:12.3f}{r['width_90']:10.2f}{r['below_alarm_bound']:20.3f}")


if __name__ == "__main__":
    main()
