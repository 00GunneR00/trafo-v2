"""Beklenmeyen ısınma alarmının yapay arızalarla değerlendirilmesi (bkz. KARARLAR.md, 4).

İki alarm kuralı kıyaslanır:
  esik:  s saatindeki gerçek OT, (s - h) anında yapılmış tahminin kalibre %99 üst
         sınırını art arda N saat aşarsa alarm.
  cusum: z = (gerçek - ortanca tahmin) / ölçek; S = max(0, S + z - k); S > H ise alarm.
         Küçük ama süren sapmaları biriktirir (Page, 1954). Ölçek, %90 aralığın yarı
         genişliğinin 1,645'e bölümü (normal dağılımda bir standart sapmaya karşılık gelir).

Yapay arıza: OT'ye 72 saat boyunca eklenen ısınma. Yük sütunlarına dokunulmaz
("yük aynı, sıcaklık beklenenden yüksek" = soğutma arızası benzeri).
  kademeli: 24 saatte tam büyüklüğe çıkar
  ani:      3 saatte tam büyüklüğe çıkar

1. val'de her kuralın parametre ızgarası denenir; haftada en fazla 1 yanlış alarm şartıyla
   3 °C kademeli arızayı en çok yakalayan seçilir.
2. Seçilen kurallar test'te raporlanır: yakalama oranı, gecikme, haftalık yanlış alarm.

Çalıştırma: python src\\anomaly.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import HORIZONS, ROOT, TARGET, TEST_END, TRAIN_END, VAL_END, load_raw
from alarm import alarm_fn, signals
from forecaster import Forecaster

FAULT_HOURS = 72
MAGNITUDES = [1.0, 2.0, 3.0, 5.0]
FAULT_TYPES = {"kademeli": 24, "ani": 3}
N_INJECTIONS = 40
N_GRID = [1, 2, 3, 4, 6, 8]
CUSUM_K = [0.25, 0.5, 1.0]
CUSUM_H = [2, 4, 6, 8, 12, 16, 24]
MAX_FALSE_PER_WEEK = 1.0
WARMUP = 48   # arızadan önce alınan saat: art arda sayım ve CUSUM arıza başında sıfırdan başlamasın
REPORTS = ROOT / "reports"


def fault_offset(magnitude: float, ramp_hours: int) -> np.ndarray:
    k = np.arange(FAULT_HOURS)
    return magnitude * np.minimum(1.0, (k + 1) / ramp_hours)


def false_alarms_per_week(alarm: np.ndarray) -> float:
    onsets = alarm & ~np.concatenate([[False], alarm[:-1]])    # alarmın başladığı saatler
    return float(onsets.sum() / (len(alarm) / (24 * 7)))


def evaluate_split(fc: Forecaster, df, start: int, end: int, rng: np.random.Generator) -> dict:
    """Bir bölüm için temiz veride sinyaller + her arıza senaryosu için sinyaller."""
    clean = signals(fc, df, start, end)
    injections = {}
    ot_col = df.columns.get_loc(TARGET)
    for ftype, ramp in FAULT_TYPES.items():
        for m in MAGNITUDES:
            offset = fault_offset(m, ramp)
            starts = rng.integers(start + WARMUP, end - FAULT_HOURS, N_INJECTIONS)
            runs = []
            for t0 in starts:
                faulty = df.copy()
                faulty.iloc[t0:t0 + FAULT_HOURS, ot_col] += offset
                runs.append(signals(fc, faulty, t0 - WARMUP, t0 + FAULT_HOURS))
            injections[(ftype, m)] = runs
    return {"clean": clean, "injections": injections}


def score(ev: dict, h_idx: int, rule: dict) -> dict:
    fn = alarm_fn(rule)
    out = {"false_per_week": false_alarms_per_week(fn(ev["clean"], h_idx))}
    for (ftype, m), runs in ev["injections"].items():
        delays = []
        for sig in runs:
            a = fn(sig, h_idx)[WARMUP:]                    # yalnızca arıza süresi
            hits = np.flatnonzero(a)
            delays.append(int(hits[0]) if len(hits) else None)
        found = [d for d in delays if d is not None]
        out[f"{ftype}_{m:g}"] = {
            "detection_rate": len(found) / len(delays),
            "median_delay_h": float(np.median(found)) if found else None,
        }
    return out


def rule_grid():
    for n in N_GRID:
        yield {"kind": "esik", "n": n}
    for k in CUSUM_K:
        for h_thr in CUSUM_H:
            yield {"kind": "cusum", "k": k, "H": h_thr}


def rule_name(rule: dict, h: int) -> str:
    if rule["kind"] == "esik":
        return f"eşik, ufuk {h} sa, art arda {rule['n']} sa"
    return f"CUSUM, ufuk {h} sa, k={rule['k']}, H={rule['H']}"


def plot_example(fc: Forecaster, df, h: int, rule: dict):
    """Test'ten tek bir 3 °C kademeli arıza örneği: gerçek, tahmin, alarm."""
    i = HORIZONS.index(h)
    t0 = VAL_END + 24 * 40
    faulty = df.copy()
    faulty.iloc[t0:t0 + FAULT_HOURS, df.columns.get_loc(TARGET)] += fault_offset(3.0, 24)
    s = np.arange(t0 - 72, t0 + FAULT_HOURS + 48)
    pred = fc.predict(faulty, s - h)
    sig = signals(fc, faulty, s[0], s[-1] + 1)
    a = alarm_fn(rule)(sig, i)
    idx = faulty.index[s]
    ot_f = faulty[TARGET].to_numpy()[s]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, df[TARGET].to_numpy()[s], color="gray", lw=1, ls="--", label="arızasız gerçek")
    ax.plot(idx, ot_f, color="C3", lw=1.5, label="arızalı gerçek (+3 °C, kademeli)")
    ax.plot(idx, pred[:, i, 1], color="C0", lw=1, label=f"{h} saat önceki tahmin (ortanca)")
    ax.fill_between(idx, fc.lower_bound(pred)[:, i], fc.upper_bound(pred, "interval")[:, i],
                    color="C0", alpha=0.12, label="%90 aralık (kalibre)")
    ax.axvspan(idx[72], idx[72 + FAULT_HOURS - 1], color="C3", alpha=0.08, label="arıza süresi")
    if a.any():
        ax.scatter(idx[a], ot_f[a], color="k", s=14, zorder=5, label="alarm")
    ax.set_ylabel("OT °C")
    ax.set_title(f"Yapay arıza örneği (test) - {rule_name(rule, h)}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "04_anomali_ornek.png", dpi=120)


def print_result(title: str, result: dict):
    print(f"\n{title}\n  yanlış alarm/hafta: {result['false_per_week']:.2f}")
    print(f"  {'arıza':16s}{'yakalama':>10s}{'ortanca gecikme (sa)':>22s}")
    for key, r in result.items():
        if key == "false_per_week":
            continue
        d = "-" if r["median_delay_h"] is None else f"{r['median_delay_h']:.0f}"
        print(f"  {key + ' °C':16s}{r['detection_rate']:10.2f}{d:>22s}")


def main():
    df = load_raw()
    fc = Forecaster()
    if not fc.calibration:
        raise SystemExit("Önce src/calibrate.py çalıştırılmalı")

    val = evaluate_split(fc, df, TRAIN_END, VAL_END, np.random.default_rng(0))
    grid = []
    for i, h in enumerate(HORIZONS):
        for rule in rule_grid():
            grid.append({"h": h, "rule": rule, **score(val, i, rule)})

    def pick(kind):
        ok = [g for g in grid if g["rule"]["kind"] == kind and g["false_per_week"] <= MAX_FALSE_PER_WEEK]
        return max(ok, key=lambda g: (g["kademeli_3"]["detection_rate"],
                                      -(g["kademeli_3"]["median_delay_h"] or 1e9)))

    test = evaluate_split(fc, df, VAL_END, TEST_END, np.random.default_rng(1))
    chosen = {}
    for kind in ["esik", "cusum"]:
        best = pick(kind)
        name = rule_name(best["rule"], best["h"])
        print(f"\n[{kind}] val'de seçilen: {name}  (val yanlış alarm/hafta {best['false_per_week']:.2f}, "
              f"3°C kademeli yakalama {best['kademeli_3']['detection_rate']:.2f})")
        result = score(test, HORIZONS.index(best["h"]), best["rule"])
        print_result(f"TEST - {name}", result)
        chosen[kind] = {"horizon_h": best["h"], "rule": best["rule"], "val": {k: v for k, v in best.items()
                        if k not in ("h", "rule")}, "test": result}

    winner = max(chosen, key=lambda k: chosen[k]["val"]["kademeli_3"]["detection_rate"])
    report = {"model_version": fc.version, "chosen": chosen, "production_rule": winner,
              "selection": {"max_false_per_week": MAX_FALSE_PER_WEEK, "target": "kademeli_3",
                            "note": "seçim val'de, rapor test'te"},
              "setup": {"fault_hours": FAULT_HOURS, "magnitudes": MAGNITUDES,
                        "fault_types_ramp_hours": FAULT_TYPES, "injections_per_case": N_INJECTIONS}}
    (REPORTS / "anomaly.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_example(fc, df, chosen[winner]["horizon_h"], chosen[winner]["rule"])
    print(f"\nAPI'de kullanılacak kural: {winner}.  Rapor: {REPORTS / 'anomaly.json'}")


if __name__ == "__main__":
    main()
