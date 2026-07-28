#!/usr/bin/env python
"""多 checkpoint 集成 + 旋转 TTA 推理脚本。

对同一含噪输入：
  1. 用多个 checkpoint（如 --metric cd 选出的 top-3）分别推理；
  2. 可选旋转/缩放 TTA：每个输入做若干随机刚体变换，推理后逆变换回原坐标系；
  3. 所有预测逐点融合（median 抗离群，契合拉普拉斯噪声；也可选 mean）。

用法示例：
    python scripts/ensemble_predict.py \
      --task configs/task/predict_vm.yaml \
      --ckpts ckpt_a.pkl ckpt_b.pkl ckpt_c.pkl \
      --tta 4 \
      --fusion median \
      --output_dir results_ensemble

    # 快速调试（只处理前 2 个模型）
    python scripts/ensemble_predict.py --task configs/task/predict_vm.yaml \
      --ckpts best.pkl --tta 1 --limit 2 --output_dir tmp_ensemble

注意：所有 checkpoint 必须与 --task 中的 model 配置同构（例如同为 VM 或同为
StraightPCF）。不同架构的集成可分两次运行本脚本再用 numpy 手动融合。
"""

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np


def load_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf

    if not path.exists():
        raise SystemExit(f"配置文件不存在: {path}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise SystemExit(f"配置文件必须是 mapping: {path}")
    return cfg


def config_path(config_dir: str, name: str) -> Path:
    path = Path(config_dir) / name
    return path if path.suffix == ".yaml" else path.with_suffix(".yaml")


def build_views(noisy: np.ndarray, num_views: int, scale_min: float, scale_max: float, rs: np.random.RandomState):
    """返回 [(R, s), ...]：视图 0 为恒等，其余为随机旋转(+可选缩放)。"""
    from scipy.spatial.transform import Rotation

    views = [(np.eye(3, dtype=np.float32), 1.0)]
    for _ in range(num_views - 1):
        R = Rotation.random(random_state=rs).as_matrix().astype(np.float32)
        s = float(rs.uniform(scale_min, scale_max)) if scale_max > scale_min else 1.0
        views.append((R, s))
    return views


def main():
    parser = argparse.ArgumentParser(description="多 checkpoint 集成 + TTA 推理")
    parser.add_argument("--task", default="configs/task/predict_vm.yaml", help="预测任务配置（决定数据与模型结构）")
    parser.add_argument("--ckpts", nargs="+", required=True, help="参与集成的 checkpoint 列表")
    parser.add_argument("--tta", type=int, default=1, help="TTA 视图数，1=不做 TTA；视图 0 恒为恒等变换")
    parser.add_argument("--tta_scale_min", type=float, default=1.0, help="TTA 随机缩放下限")
    parser.add_argument("--tta_scale_max", type=float, default=1.0, help="TTA 随机缩放上限")
    parser.add_argument("--fusion", choices=["median", "mean"], default="median", help="融合方式")
    parser.add_argument("--output_dir", default="results_ensemble", help="输出根目录")
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个模型（调试用）")
    args = parser.parse_args()

    import jittor as jt
    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    from src.data.dataset import DatasetConfig
    from src.model.parse import get_model

    task = load_yaml(Path(args.task))
    components = task.get("components")
    if not isinstance(components, dict):
        raise SystemExit(f"{args.task} 缺少 components")

    data_config = load_yaml(config_path("configs/data", components["data"]))
    transform_config = load_yaml(config_path("configs/transform", components["transform"]))
    model_config = load_yaml(config_path("configs/model", components["model"]))

    predict_cfg = data_config.get("predict_dataset")
    if predict_cfg is None:
        raise SystemExit(f"{components['data']} 中没有 predict_dataset")
    predict_dataset_config = DatasetConfig.parse(**predict_cfg).split_by_cls()

    # 每个 checkpoint 一个模型实例（共享同一 model 配置）
    models = []
    for ckpt in args.ckpts:
        if not os.path.isfile(ckpt):
            raise SystemExit(f"checkpoint 不存在: {ckpt}")
        model = get_model(
            model_config=deepcopy(model_config),
            transform_config=deepcopy(transform_config),
        )
        model.load(ckpt)
        model.set_predict(True)
        model.eval()
        models.append(model)
        print(f"loaded checkpoint: {ckpt}")

    assets = []
    for cls, ds_config in predict_dataset_config.items():
        assets.extend(ds_config.datapath.get_data())
    if args.limit is not None:
        assets = assets[: args.limit]
    print(f"共 {len(assets)} 个模型, {len(models)} 个 checkpoint, TTA 视图数 {args.tta}, 融合 {args.fusion}")

    num_predictions = len(models) * args.tta
    t_start = time.time()
    for index, lazy_asset in enumerate(assets, start=1):
        asset = lazy_asset.load()
        noisy = asset.sampled_vertices_noisy.astype(np.float32)

        rs = np.random.RandomState(args.seed + index)
        views = build_views(noisy, args.tta, args.tta_scale_min, args.tta_scale_max, rs)

        preds = []
        for R, s in views:
            x = (noisy @ R.T) * s
            for model in models:
                out = model.predict_step({"pc_noisy": jt.array(x[None])})[0]["pc_denoised"]
                out = (out / s) @ R  # 逆变换回原坐标系
                preds.append(out.astype(np.float32))

        stack = np.stack(preds, axis=0)  # (num_predictions, N, 3)
        if args.fusion == "median":
            fused = np.median(stack, axis=0)
        else:
            fused = stack.mean(axis=0)
        fused = fused.astype(np.float32)

        if fused.shape != noisy.shape:
            raise RuntimeError(f"输出 shape 错误: {fused.shape} != {noisy.shape}")
        if not np.isfinite(fused).all():
            raise RuntimeError(f"输出包含 NaN/Inf: {asset.path}")

        # 与 VMWriter 一致的输出结构：<output_dir>/<asset 相对目录>/denoised.npy
        dirname = os.path.join(args.output_dir, os.path.dirname(asset.path))
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, "denoised.npy"), fused)

        elapsed = time.time() - t_start
        eta = elapsed / index * (len(assets) - index)
        print(f"[{index}/{len(assets)}] {asset.path}  ({num_predictions} 个预测已融合, 已用 {elapsed:.0f}s, 预计剩余 {eta:.0f}s)")

    print(f"\n完成。输出目录: {args.output_dir}")
    print("提交打包: cd <output_dir>/dataset_test_noisy && zip -r ../../result.zip shapenet/")


if __name__ == "__main__":
    main()
