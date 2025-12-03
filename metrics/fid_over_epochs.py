import os
import sys
import argparse
import json
import re
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
# Delay matplotlib import to reduce memory footprint during generation
plt = None

# clean-fid for FID computation
from cleanfid import fid as clean_fid

def _quiet_cleanfid_compute(real_path: str, fake_path: str, dataset_res: int = 64) -> float:
    """Call clean-fid without emitting output to terminal."""
    import io
    import contextlib
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        return float(clean_fid.compute_fid(real_path, fake_path, mode='clean', dataset_res=dataset_res))

"""
Generate images from each epoch checkpoint and compute FID per epoch.

Assumptions:
- Checkpoints are saved under runs/<dataset>/v<version>/ckpt/checkpoint_epoch_XXX.pth
- Real images are organized in class subdirectories: real_dir/<class>/* (matching training structure)
- mediq-gan.py supports mode=generate_epoch and reads EPOCH and SAMPLES_PER_CLASS from env vars

Outputs:
- Generated images per epoch in runs/<dataset>/v<version>/generated_img_epoch/<epoch>/<class>/*.png
- A JSON with FID per epoch
- A PNG plot of FID vs epoch
"""


def list_checkpoints(run_root: Path) -> list:
    ckpt_dir = run_root / "ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    files = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    # extract epoch numbers
    epochs = []
    for f in files:
        m = re.search(r"checkpoint_epoch_(\d{3})\.pth", f.name)
        if m:
            epochs.append(int(m.group(1)))
    return epochs


def detect_structure(directory: Path):
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"Real directory '{directory}' must contain class subdirectories.")
    class_names = sorted([d.name for d in subdirs])
    return class_names


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def epoch_images_complete(gen_epoch_dir: Path, class_names: list, samples_per_class: int) -> bool:
    """Return True if every class folder under gen_epoch_dir has at least samples_per_class images."""
    if not gen_epoch_dir.exists():
        return False
    for cls in class_names:
        cls_dir = gen_epoch_dir / cls
        if _count_images(cls_dir) < samples_per_class:
            return False
    return True


def generate_for_epoch(dataset: str, version: int, epoch: int, samples_per_class: int, mediq_path: Path):
    # call mediq-gan.py with mode=generate_epoch
    env = os.environ.copy()
    env["EPOCH"] = str(epoch)
    env["SAMPLES_PER_CLASS"] = str(samples_per_class)
    cmd = [sys.executable, str(mediq_path), "--mode", "generate_epoch", "--dataset", dataset, "--version", str(version)]
    import subprocess
    res = subprocess.run(cmd, env=env, cwd=str(mediq_path.parent))
    if res.returncode != 0:
        raise RuntimeError(f"Image generation failed for epoch {epoch} (exit code {res.returncode}).")


def compute_fid_for_epoch(real_dir: Path, gen_epoch_dir: Path, dataset_res: int = 64) -> float:
    # Ensure directories exist
    if not real_dir.exists():
        raise FileNotFoundError(f"Real directory not found: {real_dir}")
    if not gen_epoch_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {gen_epoch_dir}")
    # Use clean-fid across class subdirectories; compute mean of per-class FIDs
    class_names = detect_structure(real_dir)
    fid_values = []
    for cls in class_names:
        real_cls = real_dir / cls
        fake_cls = gen_epoch_dir / cls
        if not fake_cls.exists():
            # skip missing classes
            continue
        try:
            score = _quiet_cleanfid_compute(str(real_cls), str(fake_cls), dataset_res=dataset_res)
            fid_values.append(score)
        except Exception as e:
            # skip errors per class
            print(f"[WARN] FID error for class '{cls}' at epoch {gen_epoch_dir.name}: {e}")
            continue
    if not fid_values:
        return float('nan')
    return float(np.mean(fid_values))


def compute_fid_per_class_for_epoch(real_dir: Path, gen_epoch_dir: Path, class_names: list, dataset_res: int = 64) -> dict:
    """Compute FID per class for a given epoch. Returns {class_name: fid or np.nan}."""
    out = {}
    for cls in class_names:
        real_cls = real_dir / cls
        fake_cls = gen_epoch_dir / cls
        if not fake_cls.exists():
            out[cls] = float('nan')
            continue
        try:
            score = _quiet_cleanfid_compute(str(real_cls), str(fake_cls), dataset_res=dataset_res)
            out[cls] = float(score)
        except Exception as e:
            print(f"[WARN] FID error for class '{cls}' at epoch {gen_epoch_dir.name}: {e}")
            out[cls] = float('nan')
    return out


def plot_fid(fid_by_epoch_items: list, out_path: Path):
    # Lazy import here to avoid loading pyplot during long-running loop
    global plt
    if plt is None:
        import matplotlib.pyplot as plt  # type: ignore
    epochs = [e for e, _ in sorted(fid_by_epoch_items)]
    values = [v for _, v in sorted(fid_by_epoch_items)]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('FID (clean-fid)')
    plt.title('FID per Epoch')
    plt.grid(True, alpha=0.3)
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate images per checkpoint and plot FID vs epoch")
    parser.add_argument('--dataset', type=str, required=True, choices=['ISIC2019', 'ODIR-5k', 'RetinaMNIST', 'KneeOA'])
    parser.add_argument('--version', type=int, default=0, help='Run version (v<version>)')
    parser.add_argument('--real_dir', type=str, required=True, help='Directory with real training images (class subdirs)')
    parser.add_argument('--samples_per_class', type=int, default=200, help='Images to generate per class per epoch')
    parser.add_argument('--dataset_res', type=int, default=64, help='Resolution for FID')
    parser.add_argument('--output_json', type=str, default='metrics/fid_per_epoch.json')
    parser.add_argument('--output_plot', type=str, default='metrics/fid_per_epoch.png')
    parser.add_argument('--output_plot_per_class', type=str, default='metrics/fid_per_epoch_per_class.png')
    parser.add_argument('--output_json_per_class', type=str, default='metrics/fid_per_epoch_per_class.json')
    parser.add_argument('--start_epoch', type=int, default=None,
                        help='Resume from this epoch (inclusive). If not set, processes all epochs found.')
    parser.add_argument('--force_regen', action='store_true',
                        help='Force regeneration of images even if epoch folder already has enough images per class.')
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    mediq_path = workspace / 'mediq-gan.py'

    run_root = workspace / 'runs' / args.dataset / f'v{args.version}'
    gen_root = run_root / 'generated_img_epoch'
    real_dir = Path(args.real_dir)
    class_names = detect_structure(real_dir)

    # Discover epochs
    epochs = list_checkpoints(run_root)
    if not epochs:
        raise RuntimeError(f"No checkpoints found under {run_root}/ckpt")

    # Optionally resume from a given epoch
    if args.start_epoch is not None:
        epochs = [e for e in epochs if e >= args.start_epoch]
        if not epochs:
            raise RuntimeError(f"No checkpoints at or after epoch {args.start_epoch} under {run_root}/ckpt")
        print(f"Resuming from epoch {epochs[0]} (requested start_epoch={args.start_epoch})")

    # Stream results to disk to avoid large in-memory dicts
    out_json_stream = Path(args.output_json).with_suffix('.jsonl')
    out_json_pc_stream = Path(args.output_json_per_class).with_suffix('.jsonl')
    ensure_dir(out_json_stream)
    ensure_dir(out_json_pc_stream)
    # Truncate existing stream files when starting fresh or resuming
    if out_json_stream.exists() and args.start_epoch is None:
        out_json_stream.unlink()
    if out_json_pc_stream.exists() and args.start_epoch is None:
        out_json_pc_stream.unlink()

    for epoch in tqdm(epochs, desc='Epochs'):
        # 1) Generate images for this epoch (unless already complete and not forced)
        gen_epoch_dir = gen_root / f'{epoch:03d}'
        if not(not args.force_regen and epoch_images_complete(gen_epoch_dir, class_names, args.samples_per_class)):
            generate_for_epoch(args.dataset, args.version, epoch, args.samples_per_class, mediq_path)
        # 2) Compute FID
        fid_score = compute_fid_for_epoch(real_dir, gen_epoch_dir, dataset_res=args.dataset_res)
        # Write epoch-level FID as JSONL line
        with open(out_json_stream, 'a') as f_stream:
            f_stream.write(json.dumps({"epoch": epoch, "fid": float(fid_score)}) + "\n")

        # Per-class FID (streamed)
        class_fids = compute_fid_per_class_for_epoch(real_dir, gen_epoch_dir, class_names, dataset_res=args.dataset_res)
        with open(out_json_pc_stream, 'a') as f_pc_stream:
            f_pc_stream.write(json.dumps({"epoch": epoch, "scores": {cls: (None if (v is None or np.isnan(v)) else float(v)) for cls, v in class_fids.items()}}) + "\n")

    # Save JSON
    # Aggregate stream to final JSON for convenience and plotting
    out_json = Path(args.output_json)
    ensure_dir(out_json)
    fid_items = []  # list of (epoch, fid) for plotting only
    if out_json_stream.exists():
        with open(out_json_stream, 'r') as f_stream:
            for line in f_stream:
                rec = json.loads(line)
                fid_items.append((int(rec["epoch"]), float(rec["fid"])) )
    with open(out_json, 'w') as f:
        json.dump({str(e): v for e, v in sorted(fid_items)}, f, indent=2)
    print(f"Saved FID per epoch to {out_json} (from stream {out_json_stream.name})")

    # Plot
    out_plot = Path(args.output_plot)
    plot_fid(fid_items, out_plot)
    print(f"Saved plot to {out_plot}")

    # Save per-class JSON
    out_json_pc = Path(args.output_json_per_class)
    ensure_dir(out_json_pc)
    # Aggregate per-class stream to final JSON without keeping large dicts during computation
    per_class_agg = {}
    if out_json_pc_stream.exists():
        with open(out_json_pc_stream, 'r') as f_pc_stream:
            for line in f_pc_stream:
                rec = json.loads(line)
                ep = str(rec.get("epoch"))
                scores = rec.get("scores", {})
                for cls, val in scores.items():
                    if cls not in per_class_agg:
                        per_class_agg[cls] = {}
                    per_class_agg[cls][ep] = val
    with open(out_json_pc, 'w') as f:
        json.dump(per_class_agg, f, indent=2)
    print(f"Saved per-class FID per epoch to {out_json_pc} (from stream {out_json_pc_stream.name})")

    # Plot per-class lines
    out_plot_pc = Path(args.output_plot_per_class)
    ensure_dir(out_plot_pc)
    # Lazy import plt
    global plt
    if plt is None:
        import matplotlib.pyplot as plt  # type: ignore
    plt.figure(figsize=(9, 5))
    # Build per-class series from aggregated JSON
    # Determine common sorted epoch list
    epochs_sorted = sorted({int(e) for e in (next(iter(per_class_agg.values())).keys())}) if per_class_agg else []
    for cls, scores in per_class_agg.items():
        xs = []
        ys = []
        for ep in sorted(scores.keys(), key=lambda x: int(x)):
            val = scores.get(ep, None)
            val_float = np.nan if (val is None) else float(val)
            if not np.isnan(val_float):
                xs.append(int(ep))
                ys.append(val_float)
        if xs:
            plt.plot(xs, ys, marker='o', label=str(cls))
    plt.xlabel('Epoch')
    plt.ylabel('FID (clean-fid)')
    plt.title('FID per Epoch (per class)')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(out_plot_pc)
    plt.close()
    print(f"Saved per-class plot to {out_plot_pc}")


if __name__ == '__main__':
    main()
