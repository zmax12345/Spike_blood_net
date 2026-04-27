import os
os.environ['OMP_NUM_THREADS'] = '8'

import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable, total=None, desc=None, leave=False, dynamic_ncols=True):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

    def tqdm(iterable, total=None, desc=None, leave=False, dynamic_ncols=True):
        return _TqdmFallback(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)

from dataset_robust import FlexibleBloodFlowDataset, sequence_sparse_collate
from dense_block_manager import DenseBlockManager
from model import SNN_CNN_Hybrid


def compute_velocity_predictions(model_output, d_values):
    tau_eff_patch = model_output["tau_eff_patch"]
    patches_per_sample = model_output["patches_per_sample"]
    batch_size = d_values.shape[0]

    d_patch = d_values.repeat_interleave(patches_per_sample).view(-1, 1, 1, 1)
    v_patch_map = d_patch / (tau_eff_patch + 1e-8)
    v_patch_mean = v_patch_map.mean(dim=(1, 2, 3)).view(batch_size, patches_per_sample)
    v_global_pred = v_patch_mean.mean(dim=1)

    tau_patch_mean = tau_eff_patch.mean(dim=(1, 2, 3)).view(batch_size, patches_per_sample)

    return v_global_pred, v_patch_mean, tau_patch_mean


def compute_scalar_metrics(v_true_list, v_pred_list):
    v_true = np.array(v_true_list, dtype=np.float64)
    v_pred = np.array(v_pred_list, dtype=np.float64)

    mae = np.mean(np.abs(v_true - v_pred))
    rmse = np.sqrt(np.mean((v_true - v_pred) ** 2))
    mape = np.mean(np.abs(v_true - v_pred) / np.maximum(np.abs(v_true), 1e-8)) * 100.0
    return mae, rmse, mape


def format_duration(elapsed_seconds):
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"


def format_markdown_table(title, mapping, key_name):
    lines = [f"### {title}", "", f"| {key_name} | Samples |", "| --- | ---: |"]
    if not mapping:
        lines.append("| (empty) | 0 |")
    else:
        for key, value in mapping.items():
            if isinstance(key, float):
                key_text = f"{key:.6f}"
            else:
                key_text = str(key)
            lines.append(f"| `{key_text}` | {value} |")
    lines.append("")
    return "\n".join(lines)


def write_training_report(report_path, run_info, epoch_records):
    train_ds = run_info["train_ds"]
    val_ds = run_info["val_ds"]

    lines = [
        "# Training Report",
        "",
        f"- Run timestamp: `{run_info['timestamp']}`",
        f"- Status: `{run_info['status']}`",
        f"- Device: `{run_info['device']}`",
        f"- Duration: `{format_duration(run_info['elapsed'])}`",
        f"- Best epoch: `{run_info['best_epoch']}`",
        f"- Best validation loss: `{run_info['best_val_loss']:.6f}`" if run_info["best_epoch"] >= 0 else "- Best validation loss: `N/A`",
        f"- Model weights path: `{run_info['model_weights_path']}`",
        f"- Loss curve path: `{run_info['loss_curve_path']}`",
        "",
        "## Run Config",
        "",
        f"- total_steps: `{run_info['total_steps']}`",
        f"- block_size: `{run_info['block_size']}`",
        f"- batch_size: `{run_info['batch_size']}`",
        f"- epochs: `{run_info['epochs']}`",
        f"- dt_us: `{run_info['dt_us']}`",
        f"- spatial_shape: `{run_info['spatial_shape']}`",
        f"- patch_shape: `{run_info['patch_shape']}`",
        f"- max_train_batches: `{run_info['max_train_batches']}`",
        f"- max_val_batches: `{run_info['max_val_batches']}`",
        "",
        "## Dataset Summary",
        "",
        f"- train_samples: `{len(train_ds)}`",
        f"- train_batches: `{run_info['train_batches']}`",
        f"- val_samples: `{len(val_ds)}`",
        f"- val_batches: `{run_info['val_batches']}`",
        "",
        "## Data Config",
        "",
        "### Train Env Config",
        "",
    ]

    for path, d_val in run_info["train_env_config"].items():
        lines.append(f"- `{path}` -> d=`{d_val}`")

    lines.extend(["", "### Val Env Config", ""])
    for path, d_val in run_info["val_env_config"].items():
        lines.append(f"- `{path}` -> d=`{d_val}`")

    lines.extend(
        [
            "",
            format_markdown_table("Train Samples Per Source", train_ds.source_sample_counts, "Source"),
            format_markdown_table("Train Samples Per Velocity", train_ds.velocity_sample_counts, "Velocity"),
            format_markdown_table("Val Samples Per Source", val_ds.source_sample_counts, "Source"),
            format_markdown_table("Val Samples Per Velocity", val_ds.velocity_sample_counts, "Velocity"),
            "## Epoch History",
            "",
            "| Epoch | LR | Train Batches | Val Batches | Train Samples | Val Samples | Train Loss | Val Loss | Val MAE | Val RMSE | Val MAPE | Beta Range | Tau Range | Patch Std Mean |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )

    if not epoch_records:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        for record in epoch_records:
            lines.append(
                f"| {record['epoch']} | {record['lr']:.6e} | "
                f"{record['train_batches_processed']}/{record['train_batches_available']} | "
                f"{record['val_batches_processed']}/{record['val_batches_available']} | "
                f"{record['train_samples_seen']}/{record['train_samples_available']} | "
                f"{record['val_samples_seen']}/{record['val_samples_available']} | "
                f"{record['train_loss']:.6f} | {record['val_loss']:.6f} | "
                f"{record['val_mae']:.6f} | {record['val_rmse']:.6f} | {record['val_mape']:.2f}% | "
                f"{record['beta_min']:.4f}-{record['beta_max']:.4f} | "
                f"{record['tau_min']:.6e}-{record['tau_max']:.6e} | "
                f"{record['patch_std_mean']:.6e} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- If `max_train_batches` is not `None`, each epoch only trains on a shuffled subset of the training loader.",
            "- If `max_val_batches` is not `None`, validation metrics are computed on only the first part of the validation loader and should be treated as subset metrics.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_epoch(
    model,
    data_loader,
    optimizer,
    device,
    total_steps,
    block_size,
    spatial_shape,
    patch_shape,
    epoch_idx,
    split_name,
    max_batches=None,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    all_v_true = []
    all_v_pred = []
    all_beta = []
    all_tau_patch_mean = []
    all_patch_std = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    max_batches = len(data_loader) if max_batches is None else min(max_batches, len(data_loader))
    processed_batches = 0
    progress_bar = tqdm(
        enumerate(data_loader),
        total=max_batches,
        desc=f"Epoch {epoch_idx} [{split_name}]",
        leave=False,
        dynamic_ncols=True,
    )

    with context:
        for batch_idx, (x_seq_sparse_data, y_true, d_values) in progress_bar:
            if batch_idx >= max_batches:
                break

            y_true = y_true.to(device)
            d_values = d_values.to(device)

            manager = DenseBlockManager(
                x_seq_sparse_data,
                batch_size=y_true.shape[0],
                spatial_shape=spatial_shape,
                patch_shape=patch_shape,
            )

            if is_train:
                optimizer.zero_grad()

            model_output = model(
                dataloader_or_generator=manager,
                total_steps=total_steps,
                block_size=block_size,
            )
            v_global_pred, _, tau_patch_mean = compute_velocity_predictions(model_output, d_values)
            loss = F.smooth_l1_loss(v_global_pred, y_true)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            processed_batches += 1
            epoch_loss += loss.item()
            all_v_true.extend(y_true.detach().cpu().numpy().tolist())
            all_v_pred.extend(v_global_pred.detach().cpu().numpy().tolist())
            all_beta.extend(model_output["beta"].detach().cpu().view(-1).numpy().tolist())
            all_tau_patch_mean.extend(tau_patch_mean.detach().cpu().view(-1).numpy().tolist())
            all_patch_std.extend(tau_patch_mean.detach().std(dim=1).cpu().numpy().tolist())

            beta_batch = model_output["beta"].detach().cpu().view(-1)
            tau_batch = tau_patch_mean.detach().cpu()
            v_batch = v_global_pred.detach().cpu()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                beta=f"{beta_batch.min().item():.3f}-{beta_batch.max().item():.3f}",
                tau=f"{tau_batch.min().item():.3e}-{tau_batch.max().item():.3e}",
                patch_std=f"{tau_batch.std(dim=1).mean().item():.2e}",
                v=f"{v_batch.min().item():.3f}-{v_batch.max().item():.3f}",
            )

    progress_bar.close()

    avg_loss = epoch_loss / max(processed_batches, 1)
    mae, rmse, mape = compute_scalar_metrics(all_v_true, all_v_pred)
    beta_min = min(all_beta) if all_beta else float("nan")
    beta_max = max(all_beta) if all_beta else float("nan")
    tau_min = min(all_tau_patch_mean) if all_tau_patch_mean else float("nan")
    tau_max = max(all_tau_patch_mean) if all_tau_patch_mean else float("nan")
    patch_std_mean = float(np.mean(all_patch_std)) if all_patch_std else float("nan")

    return {
        "loss": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "patch_std_mean": patch_std_mean,
        "processed_batches": processed_batches,
        "available_batches": len(data_loader),
        "num_samples": len(all_v_true),
        "available_samples": len(data_loader.dataset),
        "v_true": all_v_true,
        "v_pred": all_v_pred,
    }


def train_cross_env():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_steps = 5000
    block_size = 100
    batch_size = 2
    num_workers = 0
    spatial_shape = (100, 368)
    patch_shape = (50, 46)
    dt_us = 20
    epochs = 50
    max_train_batches = None
    max_val_batches = None

    train_env_config = {
        "/data/zm/Moshaboli/new_data/no1": 0.018938,
        "/data/zm/Moshaboli/new_data/no4": 0.01973,
        "/data/zm/Moshaboli/new_data/no2": 0.01942,
    }
    val_env_config = {
        "/data/zm/Moshaboli/new_data/no3": 0.01963,
    }

    mask_path = "/data/zm/Moshaboli/new_data/other_data/3.0_mask (2)_hot_pixel_mask.npy"
    model_weights_path = "/data/zm/Moshaboli/new_data/Model/best_blood_flow_model.pth"
    loss_curve_path = "/data/zm/Moshaboli/new_data/Loss_curve/spike_blood_loss_curve.png"
    report_dir = "/data/zm/Moshaboli/new_data/Markdown"
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"train_cross_{report_timestamp}.md")

    os.makedirs(os.path.dirname(model_weights_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    train_ds = FlexibleBloodFlowDataset(train_env_config, mask_path=mask_path, T=1, seq_len=total_steps, dt_us=dt_us)
    val_ds = FlexibleBloodFlowDataset(val_env_config, mask_path=mask_path, T=1, seq_len=total_steps, dt_us=dt_us)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sequence_sparse_collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_sparse_collate,
        num_workers=num_workers,
    )

    print(
        f"Dataset summary | "
        f"train_samples={len(train_ds)}, train_batches={len(train_loader)}, "
        f"val_samples={len(val_ds)}, val_batches={len(val_loader)}"
    )
    if max_train_batches is not None or max_val_batches is not None:
        print(
            f"Batch limits | "
            f"train={max_train_batches if max_train_batches is not None else 'all'}, "
            f"val={max_val_batches if max_val_batches is not None else 'all'}"
        )
    print(f"Training report will be saved to {report_path}")

    model = SNN_CNN_Hybrid(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epoch_records = []
    run_status = "completed"

    try:
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch} =====")

            train_stats = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                total_steps,
                block_size,
                spatial_shape,
                patch_shape,
                epoch,
                "Train",
                max_batches=max_train_batches,
            )
            val_stats = run_epoch(
                model,
                val_loader,
                None,
                device,
                total_steps,
                block_size,
                spatial_shape,
                patch_shape,
                epoch,
                "Val",
                max_batches=max_val_batches,
            )

            scheduler.step(val_stats["loss"])
            train_loss_history.append(train_stats["loss"])
            val_loss_history.append(val_stats["loss"])

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_records.append(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_batches_processed": train_stats["processed_batches"],
                    "train_batches_available": train_stats["available_batches"],
                    "val_batches_processed": val_stats["processed_batches"],
                    "val_batches_available": val_stats["available_batches"],
                    "train_samples_seen": train_stats["num_samples"],
                    "train_samples_available": train_stats["available_samples"],
                    "val_samples_seen": val_stats["num_samples"],
                    "val_samples_available": val_stats["available_samples"],
                    "train_loss": train_stats["loss"],
                    "val_loss": val_stats["loss"],
                    "val_mae": val_stats["mae"],
                    "val_rmse": val_stats["rmse"],
                    "val_mape": val_stats["mape"],
                    "beta_min": val_stats["beta_min"],
                    "beta_max": val_stats["beta_max"],
                    "tau_min": val_stats["tau_min"],
                    "tau_max": val_stats["tau_max"],
                    "patch_std_mean": val_stats["patch_std_mean"],
                }
            )

            print(
                f"Epoch {epoch} summary | "
                f"train_batches={train_stats['processed_batches']}/{train_stats['available_batches']}, "
                f"val_batches={val_stats['processed_batches']}/{val_stats['available_batches']}, "
                f"train_loss={train_stats['loss']:.6f}, val_loss={val_stats['loss']:.6f}, "
                f"val_mae={val_stats['mae']:.6f}, val_rmse={val_stats['rmse']:.6f}, "
                f"val_mape={val_stats['mape']:.2f}%, "
                f"beta_range=[{val_stats['beta_min']:.4f}, {val_stats['beta_max']:.4f}], "
                f"tau_range=[{val_stats['tau_min']:.6e}, {val_stats['tau_max']:.6e}], "
                f"patch_std_mean={val_stats['patch_std_mean']:.6e}"
            )

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), model_weights_path)
                print(f"Saved new best model to {model_weights_path}")

            write_training_report(
                report_path,
                {
                    "timestamp": report_timestamp,
                    "status": run_status,
                    "device": str(device),
                    "elapsed": time.time() - start_time,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model_weights_path": model_weights_path,
                    "loss_curve_path": loss_curve_path,
                    "total_steps": total_steps,
                    "block_size": block_size,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "dt_us": dt_us,
                    "spatial_shape": spatial_shape,
                    "patch_shape": patch_shape,
                    "max_train_batches": max_train_batches,
                    "max_val_batches": max_val_batches,
                    "train_batches": len(train_loader),
                    "val_batches": len(val_loader),
                    "train_env_config": train_env_config,
                    "val_env_config": val_env_config,
                    "train_ds": train_ds,
                    "val_ds": val_ds,
                },
                epoch_records,
            )
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("Training interrupted by user.")

    if train_loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_loss_history)), train_loss_history, label='Train Loss', color='blue', marker='o')
        plt.plot(range(len(val_loss_history)), val_loss_history, label='Validation Loss', color='red', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('SmoothL1 Loss')
        plt.title('Patch-Aggregated Weakly Supervised Training Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()

    elapsed = time.time() - start_time
    write_training_report(
        report_path,
        {
            "timestamp": report_timestamp,
            "status": run_status,
            "device": str(device),
            "elapsed": elapsed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_weights_path": model_weights_path,
            "loss_curve_path": loss_curve_path,
            "total_steps": total_steps,
            "block_size": block_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "dt_us": dt_us,
            "spatial_shape": spatial_shape,
            "patch_shape": patch_shape,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "train_env_config": train_env_config,
            "val_env_config": val_env_config,
            "train_ds": train_ds,
            "val_ds": val_ds,
        },
        epoch_records,
    )

    print("\n" + "=" * 60)
    print(f"Training finished in {format_duration(elapsed)}")
    print(f"Best epoch: {best_epoch}")
    if best_epoch >= 0:
        print(f"Best validation loss: {best_val_loss:.6f}")
    else:
        print("Best validation loss: N/A")
    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved training report to: {report_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    train_cross_env()
