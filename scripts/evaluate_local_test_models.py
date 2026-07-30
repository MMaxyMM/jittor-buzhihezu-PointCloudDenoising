#!/usr/bin/env python3
"""Run one or more denoising checkpoints on local_test and score CD/P2S.

Examples:
    python scripts/evaluate_local_test_models.py \
        --model "VM baseline" vm checkpoint_selection/best_checkpoint.pkl

    python scripts/evaluate_local_test_models.py \
        --model "VM" vm checkpoint_selection_L1/best_checkpoint.pkl \
        --model "CVM" cvm checkpoint_selection_cvm/best_checkpoint.pkl \
        --model "StraightPCF" straightpcf \
            checkpoint_selection_straightpcf/best_checkpoint.pkl

The local benchmark is expected to have this layout:
    dataset_train_pcd_disk/local_test/shapenet/<synset>/<model>/
        clean.npy
        noisy.npy
        normalization.npz

The matching mesh is read from:
    dataset_train/local_test/shapenet/<synset>/<model>/
        models/model_normalized.obj
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ModelEntry:
    label: str
    model_config: Path
    checkpoint: Path


@dataclass(frozen=True)
class EnsembleEntry:
    label: str
    model_config: Path
    checkpoints: Tuple[Path, ...]


EvaluationEntry = Union[ModelEntry, EnsembleEntry]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_model_config(value: str) -> Path:
    path = Path(value)
    if path.suffix not in {".yaml", ".yml"} and len(path.parts) == 1:
        path = Path("configs/model") / f"{value}.yaml"
    return resolve_path(str(path))


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return value


def discover_samples(
    data_root: Path,
    limit: int | None,
    datalist: Path | None = None,
) -> List[Tuple[str, Path]]:
    samples = []
    if datalist is not None:
        if not datalist.is_file():
            raise FileNotFoundError(f"local_test datalist 不存在: {datalist}")
        entries = [
            line.strip()
            for line in datalist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        noisy_paths = [data_root / entry / "noisy.npy" for entry in entries]
    else:
        noisy_paths = sorted(data_root.glob("**/noisy.npy"))

    for noisy_path in noisy_paths:
        if not noisy_path.is_file():
            raise FileNotFoundError(f"local_test 缓存缺少 noisy.npy: {noisy_path}")
        model_dir = noisy_path.parent
        key = model_dir.relative_to(data_root).as_posix()
        clean_path = model_dir / "clean.npy"
        normalization_path = model_dir / "normalization.npz"
        if not clean_path.is_file():
            raise FileNotFoundError(f"缺少 clean.npy: {clean_path}")
        if not normalization_path.is_file():
            raise FileNotFoundError(
                f"缺少 normalization.npz，无法正确计算 P2S: {normalization_path}"
            )
        samples.append((key, model_dir))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError(f"没有在 {data_root} 下找到 noisy.npy")
    return samples


def validate_cloud(points: np.ndarray, path: Path, expected_n: int | None = None) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点云必须是 (N, 3)，实际为 {points.shape}: {path}")
    if expected_n is not None and points.shape[0] != expected_n:
        raise ValueError(
            f"预测点数 {points.shape[0]} 与输入点数 {expected_n} 不一致: {path}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"点云含 NaN/Inf: {path}")
    return points.astype(np.float32, copy=False)


def normalize_reference(clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    centered = clean - center
    scale = float(np.sqrt((centered * centered).sum(axis=1)).max())
    if scale < 1e-12:
        raise ValueError("干净点云退化，无法归一化")
    return centered / scale, center, scale


def chamfer_squared(pred: np.ndarray, clean: np.ndarray) -> float:
    """Competition CD: bidirectional mean squared nearest-neighbour distance."""
    clean_norm, center, scale = normalize_reference(clean)
    pred_norm = (pred - center) / scale
    pred_to_clean = cKDTree(clean_norm).query(pred_norm, k=1, workers=1)[0]
    clean_to_pred = cKDTree(pred_norm).query(clean_norm, k=1, workers=1)[0]
    return float(np.mean(pred_to_clean**2) + np.mean(clean_to_pred**2))


def load_normalized_mesh(
    mesh_path: Path, normalization_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    import point_cloud_utils as pcu

    vertices, faces = pcu.load_mesh_vf(str(mesh_path))
    with np.load(normalization_path, allow_pickle=False) as normalization:
        if "center" not in normalization or "scale" not in normalization:
            raise ValueError(f"normalization.npz 缺少 center/scale: {normalization_path}")
        source_center = np.asarray(normalization["center"], dtype=np.float64)
        source_scale = float(normalization["scale"])
    if source_scale < 1e-12:
        raise ValueError(f"非法 normalization scale: {normalization_path}")
    vertices = (np.asarray(vertices, dtype=np.float64) - source_center) / source_scale
    return vertices, np.asarray(faces, dtype=np.int32)


def point_to_surface_squared(
    pred: np.ndarray,
    clean: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
) -> float:
    import point_cloud_utils as pcu

    _, center, scale = normalize_reference(clean)
    pred_norm = (pred - center) / scale
    mesh_norm = (mesh_vertices - center) / scale
    distances, _, _ = pcu.closest_points_on_mesh(
        pred_norm.astype(np.float32),
        mesh_norm.astype(np.float32),
        mesh_faces,
    )
    return float(np.mean(np.asarray(distances, dtype=np.float64) ** 2))


def score_metric(pred_value: float, noisy_value: float) -> float:
    if noisy_value < 1e-15:
        return 100.0 if pred_value < 1e-15 else 0.0
    return float(np.clip(100.0 * (1.0 - pred_value / noisy_value), 0.0, 100.0))


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def evaluate_prediction(key, model_dir, pred, mesh_root):
    noisy_path = model_dir / "noisy.npy"
    clean_path = model_dir / "clean.npy"
    normalization_path = model_dir / "normalization.npz"
    mesh_path = mesh_root / key / "models/model_normalized.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"缺少 P2S 所需 OBJ: {mesh_path}")
    noisy = validate_cloud(np.load(noisy_path, allow_pickle=False), noisy_path)
    clean = validate_cloud(np.load(clean_path, allow_pickle=False), clean_path)
    pred = validate_cloud(pred, Path(f"<prediction:{key}>"), noisy.shape[0])
    mesh_vertices, mesh_faces = load_normalized_mesh(mesh_path, normalization_path)
    cd_pred = chamfer_squared(pred, clean)
    cd_noisy = chamfer_squared(noisy, clean)
    p2s_pred = point_to_surface_squared(pred, clean, mesh_vertices, mesh_faces)
    p2s_noisy = point_to_surface_squared(noisy, clean, mesh_vertices, mesh_faces)
    return (key, cd_pred, cd_noisy, score_metric(cd_pred, cd_noisy),
            p2s_pred, p2s_noisy, score_metric(p2s_pred, p2s_noisy))


def entry_checkpoints(entry: EvaluationEntry) -> Tuple[Path, ...]:
    if isinstance(entry, EnsembleEntry):
        return entry.checkpoints
    return (entry.checkpoint,)


def run_and_evaluate(
    entry,
    samples,
    mesh_root,
    transform_config,
    fusion_mode,
    tta,
    tta_scale_min,
    tta_scale_max,
    ensemble_fusion,
    blend_alpha,
    alpha_buckets,
    seed,
):
    import jittor as jt
    from src.model.parse import get_model
    from scripts.ensemble_predict_2 import (
        alpha_for_sigma,
        build_views,
        estimate_noise_sigma,
    )

    model_config = deepcopy(load_yaml(entry.model_config))
    model_config["fusion_mode"] = {"mix": "weighted", "max": "best"}[fusion_mode]
    transform = load_yaml(transform_config)
    models = []
    for checkpoint in entry_checkpoints(entry):
        model = get_model(
            model_config=deepcopy(model_config),
            transform_config=deepcopy(transform),
        )
        model.load(str(checkpoint))
        model.set_predict(True)
        model.eval()
        models.append(model)

    use_ensemble = isinstance(entry, EnsembleEntry)
    num_views = tta if use_ensemble else 1
    rows = []
    start = time.time()
    for index, (key, model_dir) in enumerate(samples, start=1):
        noisy_path = model_dir / "noisy.npy"
        noisy = validate_cloud(np.load(noisy_path, allow_pickle=False), noisy_path)

        if use_ensemble:
            random_state = np.random.RandomState(seed + index)
            views = build_views(
                noisy, num_views, tta_scale_min, tta_scale_max, random_state
            )
        else:
            views = [(np.eye(3, dtype=np.float32), 1.0)]

        predictions = []
        with jt.no_grad():
            for rotation, scale in views:
                transformed = (noisy @ rotation.T) * scale
                for model in models:
                    output = model.predict_step(
                        {"pc_noisy": jt.array(transformed[None, ...])}
                    )
                    prediction = output[0]["pc_denoised"]
                    if not isinstance(prediction, np.ndarray):
                        prediction = prediction.numpy()
                    prediction = (np.asarray(prediction) / scale) @ rotation
                    predictions.append(prediction.astype(np.float32, copy=False))
                    del output, prediction

        stack = np.stack(predictions, axis=0)
        if ensemble_fusion == "median":
            denoised = np.median(stack, axis=0)
        else:
            denoised = stack.mean(axis=0)

        alpha = blend_alpha
        if use_ensemble and alpha_buckets:
            sigma = estimate_noise_sigma(noisy)
            alpha = alpha_for_sigma(sigma, alpha_buckets)
        if use_ensemble and alpha != 1.0:
            denoised = noisy + alpha * (denoised - noisy)

        denoised = validate_cloud(
            np.asarray(denoised), Path(f"<prediction:{key}>"), noisy.shape[0]
        )
        rows.append(evaluate_prediction(key, model_dir, denoised, mesh_root))
        del denoised, predictions, stack
        elapsed = time.time() - start
        remaining = elapsed / index * (len(samples) - index)
        print(
            f"\r[{entry.label}] 推理并评测 {index}/{len(samples)} "
            f"({len(models)} checkpoint × {num_views} TTA) "
            f"【已用时间 {format_duration(elapsed)} / "
            f"预计还需 {format_duration(remaining)}】",
            end="",
            flush=True,
        )
    print()
    del models
    jt.gc()
    return summarize_results(entry, rows)


def summarize_results(entry: EvaluationEntry, rows: Sequence[tuple]) -> dict:

    cd_pred = np.asarray([row[1] for row in rows], dtype=np.float64)
    cd_noisy = np.asarray([row[2] for row in rows], dtype=np.float64)
    cd_scores = np.asarray([row[3] for row in rows], dtype=np.float64)
    p2s_pred = np.asarray([row[4] for row in rows], dtype=np.float64)
    p2s_noisy = np.asarray([row[5] for row in rows], dtype=np.float64)
    p2s_scores = np.asarray([row[6] for row in rows], dtype=np.float64)
    cd_score = float(cd_scores.mean())
    p2s_score = float(p2s_scores.mean())
    return {
        "label": entry.label,
        "checkpoint": ", ".join(str(path) for path in entry_checkpoints(entry)),
        "samples": len(rows),
        "score": 0.5 * cd_score + 0.5 * p2s_score,
        "cd_score": cd_score,
        "p2s_score": p2s_score,
        "mean_cd": float(cd_pred.mean()),
        "mean_cd_noisy": float(cd_noisy.mean()),
        "mean_p2s": float(p2s_pred.mean()),
        "mean_p2s_noisy": float(p2s_noisy.mean()),
    }


def model_source(
    entry: EvaluationEntry,
    tta: int,
    tta_scale_min: float,
    tta_scale_max: float,
    ensemble_fusion: str,
    blend_alpha: float,
    alpha_bucket_spec: str,
) -> str:
    if isinstance(entry, ModelEntry):
        return f"{entry.label} | {entry.checkpoint}"
    checkpoints = ", ".join(str(path) for path in entry.checkpoints)
    settings = (
        f"fusion={ensemble_fusion},tta={tta},"
        f"scale={tta_scale_min}:{tta_scale_max},alpha={blend_alpha},"
        f"alpha_buckets={alpha_bucket_spec or '-'}"
    )
    return f"{entry.label} | ensemble({settings}) | {checkpoints}"


def read_tested_sources(result_file: Path) -> set:
    if not result_file.is_file():
        return set()
    sources = set()
    for line in result_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("模型来源") and "：" in line:
            sources.add(line.split("：", 1)[1].strip())
    return sources


def append_result(result: dict, result_file: Path, source: str) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    index = len(read_tested_sources(result_file)) + 1
    prefix = "\n" if result_file.exists() and result_file.stat().st_size else ""
    block = (
        f"{prefix}模型来源{index}：{source}\n"
        f"模型分数：score:{result['score']:.4f} ; "
        f"mean_CD:{result['mean_cd']:.8f} ; "
        f"mean_p2s:{result['mean_p2s']:.8f}\n"
    )
    with result_file.open("a", encoding="utf-8") as file:
        file.write(block)
        file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 local_test 上推理一个或多个模型并计算比赛 CD/P2S 分数"
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("LABEL", "MODEL_CONFIG", "CHECKPOINT"),
        help="可重复指定；MODEL_CONFIG 可写 vm/cvm/straightpcf 或 YAML 路径",
    )
    parser.add_argument(
        "--ensemble",
        action="append",
        nargs="+",
        metavar="VALUE",
        help=(
            "可重复指定：LABEL MODEL_CONFIG CHECKPOINT [CHECKPOINT ...]；"
            "对同构 checkpoint/TTA 预测进行融合"
        ),
    )
    parser.add_argument(
        "--data-root",
        default="dataset_train_pcd_disk/local_test",
        help="包含 clean.npy/noisy.npy/normalization.npz 的 local_test",
    )
    parser.add_argument(
        "--mesh-root",
        default="dataset_train/local_test",
        help="包含原始 OBJ 的 local_test",
    )
    parser.add_argument(
        "--datalist",
        default="dataset_train/local_test/datalist.txt",
        help="严格限定评测样本；传入空字符串才递归扫描data-root",
    )
    parser.add_argument(
        "--transform-config",
        default="configs/transform/predict.yaml",
    )
    parser.add_argument("--result-file", default="result.txt")
    parser.add_argument(
        "--fusion-mode",
        choices=("mix", "max"),
        default="max",
        help="mix=重叠patch距离加权融合；max=采用最大权重patch",
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=1,
        help="ensemble 的 TTA 视图数；1 表示不做 TTA",
    )
    parser.add_argument(
        "--tta_scale_min",
        "--tta-scale-min",
        dest="tta_scale_min",
        type=float,
        default=1.0,
        help="ensemble TTA 随机缩放下限",
    )
    parser.add_argument(
        "--tta_scale_max",
        "--tta-scale-max",
        dest="tta_scale_max",
        type=float,
        default=1.0,
        help="ensemble TTA 随机缩放上限",
    )
    parser.add_argument(
        "--fusion",
        choices=("median", "mean"),
        default="median",
        help="多个 checkpoint/TTA 预测的逐点融合方式",
    )
    parser.add_argument(
        "--blend_alpha",
        "--blend-alpha",
        dest="blend_alpha",
        type=float,
        default=1.0,
        help="ensemble 保守混合系数，1 表示不向 noisy 回退",
    )
    parser.add_argument(
        "--alpha_buckets",
        "--alpha-buckets",
        dest="alpha_buckets",
        default="",
        help="按估计噪声分桶的 alpha，例如 '0.010:0.5,0.020:0.8'",
    )
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None, help="仅调试前 N 个样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model and not args.ensemble:
        raise SystemExit("必须至少指定一个 --model 或 --ensemble")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须至少为 1")
    if args.tta < 1:
        raise SystemExit("--tta 必须至少为 1")
    if args.tta_scale_min <= 0 or args.tta_scale_max <= 0:
        raise SystemExit("TTA 缩放范围必须大于 0")
    if args.tta_scale_min > args.tta_scale_max:
        raise SystemExit("--tta_scale_min 不能大于 --tta_scale_max")
    if not 0.0 <= args.blend_alpha <= 1.0:
        raise SystemExit("--blend_alpha 必须在 [0, 1] 范围内")

    from scripts.ensemble_predict_2 import parse_alpha_buckets

    try:
        alpha_buckets = (
            parse_alpha_buckets(args.alpha_buckets) if args.alpha_buckets else []
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "--alpha_buckets 格式错误，应类似 '0.010:0.5,0.020:0.8'"
        ) from exc
    if any(threshold <= 0 for threshold, _ in alpha_buckets):
        raise SystemExit("--alpha_buckets 的噪声阈值必须大于 0")
    if any(not 0.0 <= alpha <= 1.0 for _, alpha in alpha_buckets):
        raise SystemExit("--alpha_buckets 的 alpha 必须在 [0, 1] 范围内")

    try:
        import point_cloud_utils  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "精确 P2S 需要 point-cloud-utils，请先执行："
            "python -m pip install point-cloud-utils"
        ) from exc

    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    data_root = resolve_path(args.data_root)
    mesh_root = resolve_path(args.mesh_root)
    datalist = resolve_path(args.datalist) if args.datalist else None
    result_file = resolve_path(args.result_file)
    transform_config = resolve_path(args.transform_config)
    entries: List[EvaluationEntry] = [
        ModelEntry(
            label=values[0],
            model_config=resolve_model_config(values[1]),
            checkpoint=resolve_path(values[2]),
        )
        for values in (args.model or [])
    ]
    for values in args.ensemble or []:
        if len(values) < 3:
            raise SystemExit(
                "--ensemble 至少需要 LABEL MODEL_CONFIG CHECKPOINT 三个值"
            )
        entries.append(
            EnsembleEntry(
                label=values[0],
                model_config=resolve_model_config(values[1]),
                checkpoints=tuple(resolve_path(value) for value in values[2:]),
            )
        )
    for entry in entries:
        for checkpoint in entry_checkpoints(entry):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    samples = discover_samples(data_root, args.limit, datalist)
    print(f"local_test 样本数: {len(samples)}")
    start = time.time()
    for entry in entries:
        tested_sources = read_tested_sources(result_file)
        source = model_source(
            entry,
            args.tta,
            args.tta_scale_min,
            args.tta_scale_max,
            args.fusion,
            args.blend_alpha,
            args.alpha_buckets,
        )
        if source in tested_sources:
            print(f"\n跳过已测试模型: {source}")
            continue
        print(
            f"\n模型: {entry.label}\ncheckpoint: "
            + ", ".join(str(path) for path in entry_checkpoints(entry))
        )
        result = run_and_evaluate(
            entry,
            samples,
            mesh_root,
            transform_config,
            args.fusion_mode,
            args.tta,
            args.tta_scale_min,
            args.tta_scale_max,
            args.fusion,
            args.blend_alpha,
            alpha_buckets,
            args.seed,
        )
        append_result(result, result_file, source)
        print(f"结果已追加写入: {result_file}")
        print(
            f"score={result['score']:.4f}, "
            f"CD_score={result['cd_score']:.4f}, "
            f"P2S_score={result['p2s_score']:.4f}, "
            f"mean_CD={result['mean_cd']:.8f}, "
            f"mean_P2S={result['mean_p2s']:.8f}"
        )
        print(
            f"noisy baseline: mean_CD={result['mean_cd_noisy']:.8f}, "
            f"mean_P2S={result['mean_p2s_noisy']:.8f}"
        )

    print(f"总耗时: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
