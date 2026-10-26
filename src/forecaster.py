"""Kaydedilmiş modeli yükleyip ham veriden °C tahmin üreten ortak katman.

Anomali değerlendirmesi ve API aynı kodu kullanır; eğitimdeki ön işlemeyle birebir aynıdır.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import FEATURE_COLS, HORIZONS, TARGET, TIME_COLS, WINDOW, Scaler, add_time_features, to_celsius
from model import QuantileLSTM, make_relative, sort_quantiles

MODELS = Path(__file__).resolve().parents[1] / "models"


class Forecaster:
    def __init__(self, meta_path: Path = MODELS / "latest.json", device: str | None = None):
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scaler = Scaler.from_dict(self.meta["scaler"])
        self.model = QuantileLSTM(self.meta["n_features"], self.meta["hidden"],
                                  self.meta["layers"], self.meta["dropout"]).to(self.device)
        state = torch.load(MODELS / self.meta["file"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        # Doğrulama verisiyle hesaplanan kalibrasyon (src/calibrate.py); yoksa ham aralık kullanılır.
        self.calibration = self.meta.get("calibration", {})

    @property
    def version(self) -> str:
        return self.meta["version"]

    def windows(self, df: pd.DataFrame, t_positions: np.ndarray) -> np.ndarray:
        """t_positions: tahminin yapıldığı satırların konumu (her biri >= WINDOW-1)."""
        if "hour_sin" not in df.columns:
            df = add_time_features(df)
        values = self.scaler.transform(df)[FEATURE_COLS + TIME_COLS].to_numpy(np.float32)
        idx = t_positions[:, None] - WINDOW + 1 + np.arange(WINDOW)[None, :]
        X = values[idx]
        return make_relative(X) if self.meta["relative_input"] else X

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, t_positions: np.ndarray | None = None, batch: int = 2048) -> np.ndarray:
        """°C cinsinden (n, ufuk, yüzdelik) tahmin. Varsayılan: verinin son saati için tek tahmin."""
        if t_positions is None:
            t_positions = np.array([len(df) - 1])
        X = self.windows(df, t_positions)
        outs = []
        for i in range(0, len(X), batch):
            outs.append(self.model(torch.from_numpy(X[i:i + batch]).to(self.device)).float().cpu().numpy())
        pred = sort_quantiles(np.concatenate(outs))
        base = df[TARGET].to_numpy(np.float32)[t_positions]
        return to_celsius(pred, base, self.scaler, self.meta["target"])

    def upper_bound(self, pred_c: np.ndarray, level: str = "alarm") -> np.ndarray:
        """Kalibre edilmiş üst sınır, (n, ufuk). level: "interval" (%95 üst) ya da "alarm"."""
        shift = np.array([self.calibration.get(level, {}).get(f"h{h}", 0.0) for h in HORIZONS], np.float32)
        return pred_c[:, :, 2] + shift

    def lower_bound(self, pred_c: np.ndarray) -> np.ndarray:
        shift = np.array([self.calibration.get("interval", {}).get(f"h{h}", 0.0) for h in HORIZONS], np.float32)
        return pred_c[:, :, 0] - shift
