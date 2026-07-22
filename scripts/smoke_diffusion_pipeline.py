#!/usr/bin/env python
"""Run one real cached-mesh training step through the diffusion pipeline."""

import argparse
from pathlib import Path
import sys

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datapath import CleanNpyLazyAsset
from src.data.transform import Transform
from src.model.parse import get_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default="dataset_train_pcd")
    parser.add_argument(
        "--model_config", default="configs/model/residual_diffusion.yaml"
    )
    parser.add_argument(
        "--transform_config",
        default="configs/transform/residual_diffusion.yaml",
    )
    parser.add_argument("--use_cuda", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    jt.flags.use_cuda = args.use_cuda
    clean_paths = sorted(Path(args.cache_dir).glob("shapenet/*/*/clean.npy"))
    if not clean_paths:
        raise SystemExit(f"no clean.npy files found under {args.cache_dir}")

    transform_config = OmegaConf.to_container(
        OmegaConf.load(args.transform_config), resolve=True
    )
    model_config = OmegaConf.to_container(
        OmegaConf.load(args.model_config), resolve=True
    )
    model = get_model(
        model_config=model_config,
        transform_config=transform_config,
    )
    model.set_predict(False)
    model.train()

    asset = CleanNpyLazyAsset(path=str(clean_paths[0])).load()
    transform = Transform.parse(**transform_config["train_transform"])
    transform.apply(asset)
    processed = model._process_fn([asset])[0]
    batch = {
        key: jt.array(value[None]) if isinstance(value, np.ndarray) else value
        for key, value in processed.items()
    }
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)
    loss = model.training_step(batch)["loss"]
    optimizer.step(loss)
    print(f"asset={clean_paths[0]}")
    print(f"loss={float(loss.item()):.8f}")
    print(
        "batch="
        + ", ".join(f"{key}:{tuple(value.shape)}" for key, value in batch.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
