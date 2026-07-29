"""Run the fixed 50-cloud anchored inference panel and fallback rules.

This sweep keeps one model/panel in-process and only swaps inference
hyperparameters between runs, which is much faster than spawning
``select_best_checkpoint.py`` for every combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from select_best_checkpoint import (  # noqa: E402
    build_cd_eval_panel,
    build_validation_context,
    mean,
)


def parse_floats(value):
    return [float(item) for item in value.split(",") if item]


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def _score_panel(model, panel, metric: str) -> dict:
    import jittor as jt
    from evaluate import point_to_surface_distance

    cds = []
    noisy_cds = []
    cd_scores = []
    p2s_values = []
    noisy_p2s_values = []
    p2s_scores = []
    inference_seconds = []

    model.set_predict(True)
    model.eval()
    for sample in panel:
        noisy = sample["noisy"]
        clean = sample["clean"]
        started = time.perf_counter()
        pred = model.predict_step({"pc_noisy": jt.array(noisy[None])})
        denoised = pred[0]["pc_denoised"]
        inference_seconds.append(time.perf_counter() - started)

        from select_best_checkpoint import _chamfer_distance

        cd = _chamfer_distance(denoised, clean)
        cd_noisy = _chamfer_distance(noisy, clean)
        cds.append(cd)
        noisy_cds.append(cd_noisy)
        cd_scores.append(float(np.clip(100.0 * (1.0 - cd / cd_noisy), 0.0, 100.0)))

        if metric == "official":
            p2s = point_to_surface_distance(
                denoised, sample["mesh_vertices"], sample["mesh_faces"]
            )
            noisy_p2s = point_to_surface_distance(
                noisy, sample["mesh_vertices"], sample["mesh_faces"]
            )
            p2s_score = float(np.clip(100.0 * (1.0 - p2s / noisy_p2s), 0.0, 100.0))
            p2s_values.append(p2s)
            noisy_p2s_values.append(noisy_p2s)
            p2s_scores.append(p2s_score)

    metrics = {
        "val/cd_mean": mean(cds),
        "val/noisy_cd_mean": mean(noisy_cds),
        "val/cd_score_mean": mean(cd_scores),
        "val/inference_seconds_mean": mean(inference_seconds),
        "eval/patch_batch_size": float(
            getattr(model, "inference_patch_batch_size", 0) or 0
        ),
    }
    if p2s_scores:
        metrics.update({
            "val/p2s_mean": mean(p2s_values),
            "val/noisy_p2s_mean": mean(noisy_p2s_values),
            "val/p2s_score_mean": mean(p2s_scores),
            "val/final_score": 0.5 * mean(cd_scores) + 0.5 * mean(p2s_scores),
        })
        score = metrics["val/final_score"]
    else:
        score = metrics["val/cd_mean"]
    return {"score": score, "metrics": metrics}


def configure_model(model, beta, steps, noise_std, seed, patch_batch_size):
    model.blend_beta = float(beta)
    model.num_inference_steps = int(steps)
    model.inference_noise_std = float(noise_std)
    model._current_inference_std = float(noise_std)
    model.sampling_seed = int(seed)
    model.sampling_seeds = [int(seed)]
    model.inference_patch_batch_size = int(patch_batch_size)
    # beta=0 is exact anchor fallback; disable variance gating work.
    model.use_variance_gate = float(beta) > 0.0


def aggregate_seed_results(rows):
    groups = {}
    for row in rows:
        key = (row["beta"], row["steps"], row["noise_std"])
        groups.setdefault(key, []).append(row)
    aggregated = []
    for (beta, steps, noise_std), members in groups.items():
        aggregated.append({
            "beta": beta,
            "steps": steps,
            "noise_std": noise_std,
            "mean_score": statistics.mean(row["score"] for row in members),
            "seeds": [row["seed"] for row in members],
            "runs": members,
        })
    return sorted(aggregated, key=lambda row: row["mean_score"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--task_template",
        default="configs/task/train_anchored_diffusion_stage2.yaml",
    )
    parser.add_argument("--mesh_root", default="dataset_train")
    parser.add_argument("--output_dir", default="anchored_inference_sweep")
    parser.add_argument("--betas", default="0,0.5,0.75,1")
    parser.add_argument("--steps", default="4,8")
    parser.add_argument("--noise_stds", default="0.010,0.0125,0.015")
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--full_top", type=int, default=3)
    parser.add_argument("--anchor_score", type=float, default=69.77)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    parser.add_argument("--cd_limit", type=int, default=50)
    parser.add_argument("--cd_points", type=int, default=32768)
    parser.add_argument("--noise_std_min", type=float, default=0.005)
    parser.add_argument("--noise_std_max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument(
        "--patch_batch_size",
        type=int,
        default=64,
        help="Patches per GPU micro-batch during sweep inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse finished runs recorded in quick_panel.json / full dirs.",
    )
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    if len(seeds) > 2:
        raise SystemExit("at most two deterministic seeds are allowed")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fake argparse namespace expected by select_best_checkpoint helpers.
    selection_args = SimpleNamespace(
        task_template=args.task_template,
        data_config="",
        model_override=[],
        use_cuda=args.use_cuda,
        seed=args.seed,
        metric="official",
        cd_limit=args.cd_limit,
        cd_points=args.cd_points,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        mesh_root=args.mesh_root,
        patch_batch_size=args.patch_batch_size,
    )

    import jittor as jt
    from src.model.parse import get_model

    jt.flags.use_cuda = args.use_cuda
    context = build_validation_context(selection_args)
    print(f"Building fixed official panel (limit={args.cd_limit})...")
    panel = build_cd_eval_panel(context, selection_args)
    print(f"  panel size: {len(panel)}")
    print(f"  patch_batch_size: {args.patch_batch_size}")

    model = get_model(
        model_config=deepcopy(context["model_config"]),
        transform_config=deepcopy(context["transform_config"]),
    )
    model.load(str(Path(args.checkpoint)))
    model.inference_patch_batch_size = int(args.patch_batch_size)

    existing = {}
    quick_path = output_dir / "quick_panel.json"
    if args.resume and quick_path.exists():
        existing = {
            run["name"]: run
            for run in json.loads(quick_path.read_text(encoding="utf-8")).get(
                "runs", []
            )
            if run.get("score") is not None
        }

    combinations = list(itertools.product(
        parse_floats(args.betas),
        parse_ints(args.steps),
        parse_floats(args.noise_stds),
        seeds,
    ))
    # beta=0 is bitwise/identity-ish anchor; steps/std/seed do not matter.
    # Evaluate once and broadcast to all beta=0 combinations.
    beta_zero_template = None
    quick_runs = []
    for index, (beta, steps, noise_std, seed) in enumerate(combinations, start=1):
        name = (
            f"beta-{beta:g}_steps-{steps}_std-{noise_std:g}_seed-{seed}"
        )
        if name in existing:
            print(f"[{index}/{len(combinations)}] resume {name}")
            quick_runs.append(existing[name])
            continue

        if abs(beta) < 1e-12 and beta_zero_template is not None:
            row = dict(beta_zero_template)
            row.update({
                "name": name,
                "beta": beta,
                "steps": steps,
                "noise_std": noise_std,
                "seed": seed,
            })
            print(
                f"[{index}/{len(combinations)}] reuse beta=0 for {name}: "
                f"{row['score']:.4f}"
            )
            quick_runs.append(row)
            continue

        print(
            f"[{index}/{len(combinations)}] evaluate {name} "
            f"(patch_batch={args.patch_batch_size})"
        )
        configure_model(
            model, beta, steps, noise_std, seed, args.patch_batch_size
        )
        result = _score_panel(model, panel, metric="official")
        row = {
            "name": name,
            "beta": beta,
            "steps": steps,
            "noise_std": noise_std,
            "seed": seed,
            "score": result["score"],
            "metrics": result["metrics"],
        }
        print(
            f"  score={row['score']:.4f} "
            f"infer={result['metrics'].get('val/inference_seconds_mean', 0):.3f}s/sample"
        )
        if abs(beta) < 1e-12:
            beta_zero_template = row
        quick_runs.append(row)
        (output_dir / "quick_panel.json").write_text(
            json.dumps(
                {
                    "summary": aggregate_seed_results(quick_runs),
                    "runs": quick_runs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if hasattr(jt, "gc"):
            jt.gc()

    quick = aggregate_seed_results(quick_runs)
    (output_dir / "quick_panel.json").write_text(
        json.dumps({"summary": quick, "runs": quick_runs}, indent=2),
        encoding="utf-8",
    )

    # Full validation = same panel without cd_limit (all validate clouds).
    full_args = SimpleNamespace(**vars(selection_args))
    full_args.cd_limit = None
    print("Building full official panel...")
    full_panel = build_cd_eval_panel(context, full_args)
    print(f"  full panel size: {len(full_panel)}")

    full_runs = []
    for row in quick[:args.full_top]:
        for seed in row["seeds"]:
            name = (
                f"beta-{row['beta']:g}_steps-{row['steps']}_"
                f"std-{row['noise_std']:g}_seed-{seed}_full"
            )
            print(f"full evaluate {name}")
            configure_model(
                model,
                row["beta"],
                row["steps"],
                row["noise_std"],
                seed,
                args.patch_batch_size,
            )
            result = _score_panel(model, full_panel, metric="official")
            full_runs.append({
                "name": name,
                "beta": row["beta"],
                "steps": row["steps"],
                "noise_std": row["noise_std"],
                "seed": seed,
                "score": result["score"],
                "metrics": result["metrics"],
            })
            print(f"  score={result['score']:.4f}")
            if hasattr(jt, "gc"):
                jt.gc()

    full = aggregate_seed_results(full_runs)
    fallback = not full or full[0]["mean_score"] <= args.anchor_score
    beta_one = [row["mean_score"] for row in quick if row["beta"] == 1.0]
    report = {
        "quick_best": quick[0] if quick else None,
        "full_results": full,
        "fallback_to_anchor": fallback,
        "stage1_expand_model": not (
            args.stage == "stage1"
            and beta_one
            and max(beta_one) < args.anchor_score - 2.0
        ),
        "stop_rule": ("keep_anchor" if fallback else "keep_refiner"),
        "patch_batch_size": args.patch_batch_size,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
