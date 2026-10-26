"""LSTM eğitimi, MLflow kaydı ve model dosyası.

Örnekler:
  python src\\train_lstm.py                              # varsayılan: delta hedef, göreli girdi, AMP açık
  python src\\train_lstm.py --target level --absolute    # kıyas: seviye hedef, mutlak girdi
  python src\\train_lstm.py --no-amp                     # mixed precision kapalı süre kıyası
  python src\\train_lstm.py --seed 1 --save              # modeli models/ altına kaydet
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import HORIZONS, QUANTILES, ROOT, WINDOW, prepare, to_celsius
from metrics import interval_metrics, point_metrics
from model import QuantileLSTM, make_relative, pinball_loss, sort_quantiles

MODELS = ROOT / "models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["delta", "level"], default="delta")
    p.add_argument("--absolute", action="store_true", help="OT girdisini göreli yapma")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        outs.append(model(xb).float().cpu().numpy())
    return sort_quantiles(np.concatenate(outs))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    relative = not args.absolute

    data, scaler = prepare(args.target)
    if relative:
        data = {k: (make_relative(X), y, b, t) for k, (X, y, b, t) in data.items()}
    Xtr, ytr, _, _ = data["train"]
    Xva, yva, bva, _ = data["val"]

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=args.batch, shuffle=True, drop_last=True)
    model = QuantileLSTM(Xtr.shape[2], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # AMP: ileri geçiş float16'da, ağırlıklar float32'de. GradScaler küçük gradyanların
    # float16'da sıfıra yuvarlanmasını (underflow) önlemek için kaybı ölçekler.
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    run_name = f"lstm_{args.target}_{'goreli' if relative else 'mutlak'}{'' if use_amp else '_noamp'}_s{args.seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "lstm")
        mlflow.log_params({**vars(args), "relative_input": relative, "amp": use_amp,
                           "window": WINDOW, "device": torch.cuda.get_device_name(0) if use_amp or device.type == "cuda" else "cpu"})

        best_val, best_state, bad_epochs, epoch_times = float("inf"), None, 0, []
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(xb)
                loss = pinball_loss(pred.float(), yb)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # LSTM'de patlayan gradyana karşı
                grad_scaler.step(opt)
                grad_scaler.update()
                total += loss.item() * len(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_times.append(time.time() - t0)

            val_pred = torch.from_numpy(predict(model, Xva, device))
            val_loss = pinball_loss(val_pred, torch.from_numpy(yva)).item()
            train_loss = total / (len(loader) * args.batch)
            mlflow.log_metrics({"train_pinball": train_loss, "val_pinball": val_loss,
                                "epoch_seconds": epoch_times[-1]}, step=epoch)
            print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  {epoch_times[-1]:.1f}s")

            # Erken durdurma: val kaybı `patience` epoch boyunca iyileşmezse dur, en iyi ağırlıklara dön.
            if val_loss < best_val - 1e-4:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break

        model.load_state_dict(best_state)
        metrics = {"best_val_pinball": best_val, "epochs_run": epoch + 1,
                   "mean_epoch_seconds": float(np.mean(epoch_times[1:] or epoch_times))}
        for split_name in ["val", "test"]:
            X, y, base, _ = data[split_name]
            pq_c = to_celsius(predict(model, X, device), base, scaler, args.target)
            y_c = to_celsius(y, base, scaler, args.target)
            metrics.update(point_metrics(y_c, pq_c[:, :, 1], split_name))
            metrics.update(interval_metrics(y_c, pq_c[:, :, 0], pq_c[:, :, 2], split_name))
        mlflow.log_metrics(metrics)

        print(f"\n{run_name}")
        for h in HORIZONS:
            print(f"h{h:<2d}  MAE val {metrics[f'val_mae_h{h}']:.3f} test {metrics[f'test_mae_h{h}']:.3f}"
                  f"   kapsama val {metrics[f'val_coverage_h{h}']:.2f} test {metrics[f'test_coverage_h{h}']:.2f}"
                  f"   genişlik test {metrics[f'test_width_h{h}']:.2f} °C")
        print(f"ortalama epoch süresi {metrics['mean_epoch_seconds']:.2f}s, {metrics['epochs_run']} epoch")

        if args.save:
            MODELS.mkdir(exist_ok=True)
            version = time.strftime("%Y%m%d-%H%M%S")
            path = MODELS / f"lstm_{version}.pt"
            torch.save(model.state_dict(), path)
            meta = {
                "version": version, "file": path.name, "run_name": run_name,
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "target": args.target, "relative_input": relative,
                "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                "n_features": int(Xtr.shape[2]), "window": WINDOW,
                "horizons": HORIZONS, "quantiles": QUANTILES,
                "scaler": scaler.to_dict(),
                "test_metrics": {k: v for k, v in metrics.items() if k.startswith("test_")},
            }
            (MODELS / f"lstm_{version}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            (MODELS / "latest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            mlflow.log_artifact(str(path))
            print(f"kaydedildi: {path}")


if __name__ == "__main__":
    main()
