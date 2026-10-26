"""Keşif analizi: problem tanımından önce verinin yapısını anlamak.

Çıktılar reports/ klasörüne grafik olarak yazılır, özet sayılar ekrana basılır.
Çalıştırma: .venv\\Scripts\\python.exe src\\explore.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "raw" / "ETTh1.csv", parse_dates=["date"]).set_index("date")

# ETT makalesindeki standart bölme: 12 ay eğitim, 4 ay doğrulama, 4 ay test (saat cinsinden).
TRAIN_END = 12 * 30 * 24
VAL_END = TRAIN_END + 4 * 30 * 24
TEST_END = VAL_END + 4 * 30 * 24
parts = {
    "train": df.iloc[:TRAIN_END],
    "val": df.iloc[TRAIN_END:VAL_END],
    "test": df.iloc[VAL_END:TEST_END],
}

print("== Bölümler ==")
for name, part in parts.items():
    print(f"{name:5s} {part.index.min():%Y-%m-%d} -> {part.index.max():%Y-%m-%d}  "
          f"n={len(part):5d}  OT ort={part.OT.mean():6.2f}  std={part.OT.std():5.2f}")

# 1) OT'nin tamamı, bölümler renkli
fig, ax = plt.subplots(figsize=(14, 4))
for name, part in parts.items():
    ax.plot(part.index, part.OT, lw=0.5, label=name)
ax.set_title("ETTh1 - Yağ sıcaklığı (OT), bölümlere göre")
ax.set_ylabel("°C")
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "01_ot_tum.png", dpi=120)

# 2) Otokorelasyon: OT kaç saat önceki değerine ne kadar benziyor?
train_ot = parts["train"].OT
lags = list(range(1, 24 * 8))
acf = [train_ot.autocorr(l) for l in lags]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(lags, acf)
for d in range(1, 8):
    ax.axvline(24 * d, color="gray", lw=0.5, ls="--")
ax.set_title("OT otokorelasyonu (train) - kesikli çizgiler gün sınırları")
ax.set_xlabel("gecikme (saat)")
fig.tight_layout()
fig.savefig(REPORTS / "02_otokorelasyon.png", dpi=120)
print("\n== Otokorelasyon (train) ==")
for l in [1, 6, 12, 24, 48, 168]:
    print(f"lag {l:3d} saat: {train_ot.autocorr(l):.3f}")

# 3) Günlük döngü: saatlere göre ortalama OT ve yük
hourly = parts["train"].groupby(parts["train"].index.hour).mean()
fig, ax1 = plt.subplots(figsize=(8, 3.5))
ax1.plot(hourly.index, hourly.OT, color="C3", label="OT")
ax1.set_ylabel("OT °C")
ax2 = ax1.twinx()
ax2.plot(hourly.index, hourly.HUFL, color="C0", label="HUFL (yük)")
ax2.set_ylabel("HUFL")
ax1.set_title("Günlük döngü (train ortalaması)")
ax1.set_xlabel("saat")
fig.tight_layout()
fig.savefig(REPORTS / "03_gunluk_dongu.png", dpi=120)

# 4) Yük değişkenlerinin OT ile ilişkisi
print("\n== Korelasyon (train) OT ile ==")
print(parts["train"].corr()["OT"].drop("OT").round(3).to_string())

# 5) 24 saat sonraki değişim ne kadar büyük? Problemin zorluğu buna bağlı.
delta24 = train_ot.shift(-24) - train_ot
print("\n== 24 saatlik değişim |OT(t+24) - OT(t)| (train) ==")
print(delta24.abs().describe().round(2).to_string())

# 6) Alarm eşiği adayları: train dağılımının üst yüzdelikleri ve diğer bölümlerde aşım oranı
print("\n== Alarm eşiği adayları ==")
for q in [0.90, 0.95, 0.99]:
    thr = train_ot.quantile(q)
    rates = {n: (p.OT > thr).mean() for n, p in parts.items()}
    print(f"train %{int(q*100)} = {thr:5.2f} °C  -> aşım oranı " +
          "  ".join(f"{n}={r:.1%}" for n, r in rates.items()))

# 7) Saf (naive) tahminlerin hatası: model olmadan ne kadar iyi tahmin edilebiliyor?
# Ufuk 24 saat olunca "son değer" ile "dünün aynı saati" aynı tahmine denk gelir.
print("\n== Saf tahmin hatası (val / test MAE, °C) ==")
for h in [1, 6, 24]:
    row = []
    for name in ["val", "test"]:
        s = parts[name].OT
        y = s.shift(-h).dropna()
        row.append(f"{name}={np.abs(y - s.loc[y.index]).mean():.3f}")
    print(f"ufuk {h:2d} saat, son değer: " + "  ".join(row))

print("\nGrafikler:", REPORTS)
