"""Trafo yağ sıcaklığı tahmin ve alarm servisi.

Uç noktalar:
  GET  /health        servis ve model durumu
  GET  /model         model kartı: sürüm, test ölçümleri, kalibrasyon, alarm kuralı
  POST /predict       son >=168 saatlik ölçümden 1/6/24 saat ileri aralıklı tahmin
  POST /monitor       ölçüm serisi üzerinde beklenmeyen ısınma (CUSUM) alarmı
  GET  /demo/series   panel için: ETTh1 test döneminden bir kesit, isteğe bağlı yapay arızayla
  GET  /              panel

Çalıştırma (yerel): .venv\\Scripts\\uvicorn api.main:app --reload
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alarm import cusum_path, signals  # noqa: E402
from data import HORIZONS, LOAD_COLS, TARGET, TEST_END, VAL_END, WINDOW, load_raw  # noqa: E402
from forecaster import Forecaster  # noqa: E402

app = FastAPI(title="Trafo OT tahmin ve alarm servisi", version="1.0")

fc = Forecaster(device="cpu")
anomaly_report = json.loads((ROOT / "reports" / "anomaly.json").read_text(encoding="utf-8"))
RULE = anomaly_report["chosen"]["cusum"]
RULE_H, RULE_K, RULE_THR = RULE["horizon_h"], RULE["rule"]["k"], RULE["rule"]["H"]
DEMO_DF = load_raw()
STARTED = time.time()


class Reading(BaseModel):
    date: datetime
    HUFL: float
    HULL: float
    MUFL: float
    MULL: float
    LUFL: float
    LULL: float
    OT: float


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(..., description=f"En az {WINDOW} saatlik, saat başı, ardışık ölçüm")
    threshold_c: float | None = Field(None, description="İsteğe bağlı operasyon eşiği (°C)")


class MonitorRequest(BaseModel):
    readings: list[Reading]


def to_frame(readings: list[Reading], min_rows: int) -> pd.DataFrame:
    if len(readings) < min_rows:
        raise HTTPException(422, f"En az {min_rows} saatlik ölçüm gerekli, gelen: {len(readings)}")
    df = pd.DataFrame([r.model_dump() for r in readings]).set_index("date").sort_index()
    steps = df.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1):
        raise HTTPException(422, "Ölçümler saat başı ve boşluksuz olmalı")
    return df[LOAD_COLS + [TARGET]]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": fc.version, "device": str(fc.device),
            "uptime_s": round(time.time() - STARTED)}


@app.get("/model")
def model_card():
    m = fc.meta
    return {
        "version": m["version"], "mlflow_run_id": m["mlflow_run_id"],
        "architecture": f"LSTM {m['layers']}x{m['hidden']}, girdi {m['window']} saat, hedef: {m['target']}",
        "horizons_h": m["horizons"], "quantiles": m["quantiles"],
        "test_metrics": m["test_metrics"],
        "calibration": m.get("calibration"),
        "alarm_rule": {"kind": "cusum", "horizon_h": RULE_H, "k": RULE_K, "H": RULE_THR,
                       "test": RULE["test"],
                       "note": "Yapay arızalarla ölçüldü; gerçek arıza yakalama oranı değildir."},
    }


@app.post("/predict")
def predict(req: PredictRequest):
    df = to_frame(req.readings, WINDOW)
    pred = fc.predict(df)                                    # (1, H, Q)
    lower, upper = fc.lower_bound(pred)[0], fc.upper_bound(pred, "interval")[0]
    now = df.index[-1]
    forecasts = []
    for i, h in enumerate(HORIZONS):
        item = {"horizon_h": h, "time": (now + pd.Timedelta(hours=h)).isoformat(),
                "median_c": round(float(pred[0, i, 1]), 2),
                "lower90_c": round(float(lower[i]), 2), "upper90_c": round(float(upper[i]), 2)}
        if req.threshold_c is not None:
            # Operasyon uyarısı: ölçülmüş bir model çıktısı değil, kullanıcının eşiği.
            item["threshold_risk"] = ("yüksek" if item["median_c"] > req.threshold_c else
                                      "olası" if item["upper90_c"] > req.threshold_c else "düşük")
        forecasts.append(item)
    return {"model_version": fc.version, "as_of": now.isoformat(), "current_c": float(df[TARGET].iloc[-1]),
            "forecasts": forecasts}


def monitor_frame(df: pd.DataFrame, start: int) -> dict:
    sig = signals(fc, df, start, len(df))
    i = HORIZONS.index(RULE_H)
    s_path, alarm = cusum_path(sig["z"][:, i], RULE_K, RULE_THR)
    return {"sig": sig, "i": i, "cusum": s_path, "alarm": alarm}


@app.post("/monitor")
def monitor(req: MonitorRequest):
    first = WINDOW - 1 + max(HORIZONS)        # ilk saatin (s - h) tahmini için yeterli geçmiş
    df = to_frame(req.readings, first + 1)
    m = monitor_frame(df, first)
    times = df.index[first:]
    alarms = [t.isoformat() for t, a in zip(times, m["alarm"]) if a]
    return {"model_version": fc.version,
            "rule": {"kind": "cusum", "horizon_h": RULE_H, "k": RULE_K, "H": RULE_THR},
            "hours_checked": len(times), "alarms": alarms,
            "last": {"time": times[-1].isoformat(), "cusum": round(float(m["cusum"][-1]), 3),
                     "expected_c": round(float(m["sig"]["median"][-1, m["i"]]), 2),
                     "actual_c": float(df[TARGET].iloc[-1])}}


@app.get("/demo/series")
def demo_series(day: int = Query(30, ge=0, le=110, description="Test döneminin kaçıncı günü"),
                fault_c: float = Query(0.0, ge=0, le=10),
                fault_ramp_h: int = Query(24, ge=1, le=72),
                horizon_h: int = Query(6)):
    if horizon_h not in HORIZONS:
        raise HTTPException(422, f"horizon_h şunlardan biri olmalı: {HORIZONS}")
    show = 24 * 7
    s0 = VAL_END + 24 * day
    s1 = min(s0 + show, TEST_END)
    fault_start = s0 + 48
    df = DEMO_DF.iloc[:s1].copy()
    if fault_c > 0:
        k = np.arange(min(72, s1 - fault_start))
        df.iloc[fault_start:fault_start + len(k), df.columns.get_loc(TARGET)] += fault_c * np.minimum(1.0, (k + 1) / fault_ramp_h)
    # CUSUM'u kesitten 48 saat önce başlat ki kesit başında sıfırdan başlamış olmasın
    m = monitor_frame(df, s0 - 48)
    i, cut = HORIZONS.index(horizon_h), slice(48, None)
    sig = m["sig"]
    return {
        "time": [t.isoformat() for t in df.index[s0:s1]],
        "actual": df[TARGET].iloc[s0:s1].round(3).tolist(),
        "actual_clean": DEMO_DF[TARGET].iloc[s0:s1].round(3).tolist(),
        "median": np.round(sig["median"][cut, i], 3).tolist(),
        "lower": np.round(sig["lower"][cut, i], 3).tolist(),
        "upper": np.round(sig["upper"][cut, i], 3).tolist(),
        "cusum": np.round(m["cusum"][cut], 3).tolist(),
        "alarm": m["alarm"][cut].tolist(),
        "cusum_threshold": RULE_THR, "rule_horizon_h": RULE_H, "horizon_h": horizon_h,
        "fault": {"start": df.index[fault_start].isoformat(), "hours": 72} if fault_c > 0 else None,
    }


@app.get("/")
def panel():
    return FileResponse(ROOT / "api" / "panel.html")
