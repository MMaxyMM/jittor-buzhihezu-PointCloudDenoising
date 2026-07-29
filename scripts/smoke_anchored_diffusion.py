"""RTX 4090 smoke presets for the anchored diffusion training path."""

import argparse
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

from src.model.parse import get_model


PRESET_STEPS = {
    "batch": 1,
    "100-batches": 100,
    "epoch": 1250,  # 10,000 samples / batch 8
}


def gpu_snapshot():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip().splitlines()[0]
        name, total, used = [item.strip() for item in output.split(",")]
        return {
            "name": name,
            "memory_total_mib": float(total),
            "memory_used_mib": float(used),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor_ckpt", required=True)
    parser.add_argument(
        "--preset", choices=PRESET_STEPS, default="batch"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument(
        "--model_config",
        default="configs/model/anchored_diffusion_stage1.yaml",
    )
    parser.add_argument("--use_cuda", type=int, default=1)
    args = parser.parse_args()

    jt.flags.use_cuda = args.use_cuda
    config = OmegaConf.to_container(
        OmegaConf.load(args.model_config), resolve=True
    )
    config["anchor_ckpt"] = args.anchor_ckpt
    config["patch_size"] = args.patch_size
    model = get_model(deepcopy(config), transform_config={})
    model.train()
    optimizer = jt.optim.Adam(model.get_optim_parameters(), lr=2e-4)

    rng = np.random.RandomState(123)
    gpu_before = gpu_snapshot()
    started = time.perf_counter()
    losses = []
    steps = PRESET_STEPS[args.preset]
    for _ in range(steps):
        clean = rng.normal(
            size=(args.batch_size, args.patch_size, 3)
        ).astype(np.float32) * 0.2
        noise_std = rng.uniform(
            0.005, 0.020, size=(args.batch_size, 1, 1)
        ).astype(np.float32)
        noise = rng.laplace(
            size=clean.shape
        ).astype(np.float32) * noise_std / np.sqrt(2.0)
        normals = rng.normal(size=clean.shape).astype(np.float32)
        normals /= np.maximum(
            np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8
        )
        batch = {
            "pc_noisy": jt.array(clean + noise).unsqueeze(1),
            "pc_clean": jt.array(clean).unsqueeze(1),
            "clean_normals": jt.array(normals).unsqueeze(1),
            "noise_std": jt.array(noise_std).unsqueeze(1),
        }
        loss = model.training_step(batch)["loss"]
        optimizer.step(loss)
        losses.append(float(loss.item()))

    elapsed = time.perf_counter() - started
    gpu_after = gpu_snapshot()
    result = {
        "preset": args.preset,
        "steps": steps,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "seconds": elapsed,
        "samples_per_second": steps * args.batch_size / elapsed,
        "last_loss": losses[-1],
        "finite": bool(np.isfinite(losses).all()),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }
    output = Path("smoke_anchored_diffusion.json")
    output.write_text(
        __import__("json").dumps(result, indent=2), encoding="utf-8"
    )
    print(result)
    if not result["finite"]:
        raise SystemExit("non-finite smoke loss")


if __name__ == "__main__":
    main()
