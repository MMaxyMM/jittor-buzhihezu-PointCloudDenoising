# 基于 Jittor 的点云降噪 Baseline

本项目用于点云降噪任务：输入含噪点云 `noisy.npy`，模型预测每个点的三维位移，并输出相同点数的 `denoised.npy`。项目保留官方 OBJ 训练流程，同时提供 clean point cloud 缓存训练流程，用于减少每个 epoch 重复解析 OBJ 和 mesh 表面采样造成的 CPU/IO 开销。

## 拉普拉斯噪声适配说明

本项目针对拉普拉斯噪声做了三处适配：

1. **噪声建模**(`src/data/augment.py` 的 `AugmentAddNoise`)：通过 `noise_type` 支持 `laplace`（默认）与 `gaussian`。**拉普拉斯采样与官方 starter code 完全一致**——配置值直接作为 `np.random.laplace` 的尺度参数 b。不要擅自换算 `b = std/sqrt(2)`：官方测试集生成器与 starter code 同构，换算会导致训练噪声比测试噪声小 sqrt(2) 倍，本地指标好看但提交分数下降。
2. **损失函数**(`src/model/vm.py`、`src/model/straightpcf.py`)：拉普拉斯噪声的最大似然估计对应 L1 损失，因此三个阶段（VM / CVM / DistanceModule）统一使用 Charbonnier（平滑 L1）损失 `sqrt(||d||^2 + eps)`，既对拉普拉斯重尾离群噪声鲁棒，又避免 L1 在零点不可导。
3. **推理融合**(`src/model/vm.py` 的 `patch_based_denoise`)：通过模型配置的 `fusion_mode` 选择：
   - `weighted`：每点由覆盖它的所有 patch 预测按 `exp(-dist)` 加权融合，抗离群预测，scatter 向量化实现；
   - `best`（默认）：starter code 原版的单最佳 patch 策略（向量化重实现，语义一致），更保边缘，并在本地 VM/StraightPCF 对照中同时改善 CD 和 P2S。

## 竞赛提分工具

### 按 CD 选 checkpoint

`select_best_checkpoint.py` 新增 `--metric cd`：在验证集上动态加噪、实际推理并计算 Chamfer Distance，与竞赛评分直接对齐（噪声种子固定，所有 checkpoint 输入一致）：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --task_template configs/task/train_vm_cached.yaml \
  --metric cd --cd_limit 50 \
  --output_dir checkpoint_selection_cd \
  --copy_best
```

`--noise_std_min/max` 控制 CD 评估的加噪范围，默认与训练一致（0.005~0.020）。

### 估计测试集噪声水平

噪声统计已经整合到 `precompute_clean_points.py`：

```bash
python scripts/precompute_clean_points.py \
  --mode estimate-noise \
  --noisy_input_dir dataset_test_noisy \
  --workers 16 \
  --noise_report noise_level_report.json
```

它会检查每个输入是否为 `(50000, 3) float32`，再通过局部 PCA 法向残差估计噪声水平。这个数值受曲率、点密度和 KNN 邻域影响，是间接估计，不等同于数据生成时的真实噪声参数。旧的 `scripts/estimate_noise_level.py` 仍可单独使用。

### 推理多轮迭代

模型配置（如 `configs/model/vm.yaml`）新增可选项：

```yaml
predict_rounds: 2        # 多轮迭代降噪，默认 1；>1 需在验证集确认不过度收缩
```

## 环境安装

推荐使用 Python 3.9，并确保 GCC/G++ 版本不高于 10。

```bash
conda create -n jittor2A python=3.9 -y
conda activate jittor2A
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
```

`requirements.txt` 包含：

- `jittor`
- `numpy`
- `trimesh`
- `scipy`
- `omegaconf`
- `point-cloud-utils`
- `plotly`

`point-cloud-utils` 用于生成固定 local_test 时计算精确 P2S，也用于 `evaluate.py` 和 checkpoint 综合指标；`plotly` 用于生成可交互的 HTML 点云可视化。

### 多 worker 训练时限制 CPU 线程

当 DataLoader 使用较多 `num_workers` 时，NumPy/BLAS 可能让每个 worker 再创建多个计算线程，造成 CPU 过度订阅。先在当前终端中加载脚本：

```bash
source scripts/run_single_thread.sh
```

然后继续使用原来的训练命令，例如：

```bash
python run.py --task configs/task/train_cvm_cached.yaml
```

脚本只为当前终端设置以下变量，不会自动启动训练，也不会修改配置文件：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

必须使用 `source`（或 `. scripts/run_single_thread.sh`），直接执行脚本无法修改当前终端的环境变量。关闭终端后设置会自动失效。

## 数据准备

将官方训练集和测试集放在项目根目录：

```text
dataset_train/
└── shapenet/<synset_id>/<model_id>/models/model_normalized.obj

dataset_test_noisy/
└── shapenet/<synset_id>/<model_id>/noisy.npy
```

例如：

```bash
tar xzf dataset_train.tar.gz
unzip dataset_test_noisy.zip
```

`datalist/train.txt`、`datalist/validate.txt` 和 `datalist/test.txt` 中保存相对于数据集根目录的模型路径，例如：

```text
shapenet/04401088/d7ed512f7a7daf63772afc88105fa679
```

## 原始 OBJ 训练

原始 baseline 每次读取 OBJ，并动态执行 mesh 表面采样、归一化、加噪和 patch 构造：

```bash
python run.py --task configs/task/train_vm.yaml
```

权重默认保存在：

```text
experiments/vm/checkpoint_<epoch>.pkl
```

## 划分后的缓存与固定本地测试集

### 使用 `create_local_holdout.py` 划分数据

`scripts/create_local_holdout.py` 按 ShapeNet 类别进行确定性分层划分。默认从每个类别抽取 2% 作为 `local_test`，向上取整且每类至少保留 2 个测试模型，其余模型进入 `local_train`。脚本只创建指向 `dataset_train/shapenet` 原始模型目录的相对软链接，不复制或移动 OBJ。

首次划分：

```bash
python scripts/create_local_holdout.py \
  --dataset_dir dataset_train
```

如果 `local_train` 或 `local_test` 已经存在，脚本默认拒绝覆盖。需要按照当前默认比例重新划分时使用：

```bash
python scripts/create_local_holdout.py \
  --dataset_dir dataset_train \
  --overwrite
```

`--overwrite` 只删除并重建以下两个软链接目录：

```text
dataset_train/local_train
dataset_train/local_test
```

它不会删除 `dataset_train/shapenet` 中的原始 OBJ。划分完成后会生成：

```text
dataset_train/local_train/datalist.txt
dataset_train/local_test/datalist.txt
dataset_train/local_split_manifest.json
```

当前完整数据集共 15833 个模型，使用默认的 2%、每类至少 2 个和 `seed=123` 时，预期为：

```text
local_train: 15509
local_test:    324
```

检查实际数量：

```bash
wc -l dataset_train/local_train/datalist.txt
wc -l dataset_train/local_test/datalist.txt
```

也可以显式调整比例、每类最少测试数和随机种子：

```bash
python scripts/create_local_holdout.py \
  --dataset_dir dataset_train \
  --test_ratio 0.02 \
  --min_test_per_category 2 \
  --seed 123 \
  --overwrite
```

重新划分只会更新软链接和 datalist，不会自动迁移或删除 `dataset_train_pcd_disk` 中的旧缓存。重新划分后应根据新的 datalist 补建 `local_train` 缓存，并确保本地评测只读取新的 `local_test` 清单。

### 输入目录

新的 `precompute_clean_points.py` 同时处理已经划分好的两个集合：

```text
dataset_train/
├── local_train/
│   ├── datalist.txt
│   └── shapenet/<synset_id>/<model_id> -> 原始模型目录
├── local_test/
│   ├── datalist.txt
│   └── shapenet/<synset_id>/<model_id> -> 原始模型目录
└── shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

`local_train` 和 `local_test` 可以包含真实模型目录，也可以使用相对软链接；脚本会优先读取 split 内的路径，并兼容回退到原始 `dataset_train/shapenet/...`。两个 `datalist.txt` 中每行格式仍为：

```text
shapenet/<synset_id>/<model_id>
```

### 输出结构和用途

默认命令：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd_disk \
  --workers 16 \
  --seed 123
```

输出结构：

```text
dataset_train_pcd_disk/
├── local_train/
│   └── shapenet/<synset_id>/<model_id>/
│       ├── clean.npy          # (200000, 3) float32 表面点池
│       └── vertices.npy       # (V, 3) float32，OBJ 全部原始顶点
├── local_test/
│   └── shapenet/<synset_id>/<model_id>/
│       ├── clean.npy          # 固定 (50000, 3) float32
│       ├── noisy.npy          # 固定 (50000, 3) float32
│       └── normalization.npz  # center 和 scale
├── precompute_manifest.json
└── local_test/generation_report.json
```

两部分职责不同：

- `local_train` 只缓存干净表面点和 OBJ 原始顶点。训练 transform 每次随机保留最多 1024 个原始顶点，再从 200000 点表面池补齐到 32768 点，然后重新归一化、动态加噪和构造 patch；训练噪声不会被固定。
- `local_test` 一次性生成固定的 50000 点 clean/noisy 对，并记录归一化参数。固定随机种子保证不同 checkpoint 面对完全相同的数据。当前训练和 checkpoint 选择配置暂不读取这个目录，它只作为后续本地统一评测数据保留。

local_test 默认复用 `generate_local_test_benchmark.py` 的 CD/P2S 标定逻辑，目标为 `mean_CD_noisy=0.000246`、`mean_P2S_noisy=0.000196`。`--calibration_limit 0` 表示用完整 local_test 标定；如需先降低标定时间，可使用：

```bash
python scripts/precompute_clean_points.py \
  --workers 16 \
  --calibration_limit 100
```

### 小规模冒烟测试

请使用独立输出目录，避免部分测试结果混入正式缓存：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir /tmp/dataset_train_pcd_smoke \
  --train_num_points 200000 \
  --test_num_points 50000 \
  --calibration_limit 1 \
  --workers 1 \
  --limit 1
```

如果已经知道标定得到的 `noise_scale`，可以用 `--noise_scale` 跳过重复标定：

```bash
python scripts/precompute_clean_points.py \
  --output_dir /tmp/dataset_train_pcd_smoke \
  --limit 1 \
  --workers 1 \
  --noise_scale 1.0
```

### 断点继续、覆盖与参数

有效的 local_train 缓存和完整固定的 local_test 默认会跳过，因此中断后可以重跑同一命令。若 local_test 目录只生成了一部分，为避免混合不同标定参数，脚本会要求使用 `--overwrite` 统一重建：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd_disk \
  --workers 16 \
  --overwrite
```

常用参数：

- `--train_num_points`：local_train 每个 mesh 的表面点池大小，默认 200000。
- `--test_num_points`：local_test 固定点数，默认 50000。
- `--num_vertex_samples`：local_test 最多保留的原始 OBJ 顶点数，默认 1024。
- `--workers`：并行进程数。
- `--seed`：稳定的全局随机种子。
- `--calibration_limit`：参与 local_test 噪声标定的模型数；0 表示全部。
- `--noise_scale`：直接指定标定尺度并跳过标定。
- `--limit`：每个 split 最多处理 N 个模型，仅用于测试。
- `--overwrite`：重新生成目标输出。
- `--splits local_train`：只生成训练缓存，不生成固定 local_test。
- `--splits local_test`：只生成固定 local_test。
- `--num_points`：兼容旧命令，同时覆盖两个 split 的点数；新用法建议分别设置 train/test 点数。

只准备训练缓存时，推荐：

```bash
python scripts/precompute_clean_points.py \
  --splits local_train \
  --workers 16
```

这条命令不需要执行 local_test 的 P2S 标定。

### 检查生成结果

```bash
find dataset_train_pcd_disk/local_train -name clean.npy -type f | wc -l
find dataset_train_pcd_disk/local_train -name vertices.npy -type f | wc -l
find dataset_train_pcd_disk/local_test -name clean.npy -type f | wc -l
find dataset_train_pcd_disk/local_test -name noisy.npy -type f | wc -l
find dataset_train_pcd_disk/local_test -name normalization.npz -type f | wc -l
```

当前默认划分预期 local_train 为 15509 个模型，local_test 为 324 个模型。

### 内存盘说明

不建议把默认 200000 点的完整缓存长期只放在 `/dev/shm`，因为服务器重启后 tmpfs 内容会消失。默认输出使用持久磁盘 `dataset_train_pcd_disk`。如果需要运行时加速，可以在开机后将该目录复制到内存盘并建立软链接，但 cached 配置最终必须能解析到：

```text
dataset_train_pcd_disk/local_train/shapenet/<synset>/<model>/clean.npy
```

## 使用缓存训练

`configs/data/train_cached.yaml` 已指向 `dataset_train_pcd_disk/local_train`。正式缓存训练每个 epoch 使用 10000 个训练样本，原有任务命令不变：

配置继续复用原来的 `datalist/train.txt` 和 `datalist/validate.txt`，并通过文件存在性检查只保留已经生成 local_train 缓存的条目。固定 local_test 模型不会进入现有训练或验证流程。如果重新划分后缓存尚未补齐，启动时会报告缺少的 `clean.npy`；应先补齐缓存，而不是设置 `ignore_check: True`。

```bash
python run.py --task configs/task/train_vm_cached.yaml
```

缓存模式的数据流程为：

```text
dataset_train_pcd_disk/local_train 中的 vertices.npy + clean.npy
  -> 随机取最多 1024 个原始顶点
  -> 从表面点池补齐到 32768 点
  -> 归一化
  -> 动态添加 Laplace 噪声
  -> 构造 1000 点局部 patch
  -> 训练 displacement/velocity target
```

原始 OBJ 配置没有被覆盖，仍可随时使用：

```bash
python run.py --task configs/task/train_vm.yaml
```

## 选择最佳 Checkpoint

### `select_best_checkpoint.py` 的作用

训练会生成多个 `checkpoint_<epoch>.pkl`。`select_best_checkpoint.py` 支持 `loss`、`cd` 和 `composite` 三种模式。`composite` 使用 loss/CD/P2S 的加权排名，默认权重为 `1:2:2`；配合 `--prefilter_top_k 10` 时，会先对最后 70 个 checkpoint 做快速 CD 初筛，再对前 10 名完整计算 loss/CD/P2S。精确 P2S 需要 local_train 对应的原始 OBJ。所有 checkpoint 使用固定随机种子，输入保持一致。

### 快速测试

仅评估前 3 个 checkpoint：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --limit 3
```

### 评估全部 checkpoint 并复制最佳权重

原始 OBJ 训练对应：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --task_template configs/task/train_vm.yaml \
  --output_dir checkpoint_selection \
  --copy_best
```

缓存训练对应：

```bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments/vm_L1_2.0 \
  --task_template configs/task/train_vm_cached.yaml \
  --mesh_dir dataset_train/local_train \
  --output_dir checkpoint_selection_L1_2.0 \
  --prefilter_top_k 10 \
  --copy_best
```

输出包括：

```text
checkpoint_selection/
├── checkpoint_ranking.csv
├── checkpoint_ranking.json
├── best_checkpoint.pkl
└── logs/
```

常用参数：

- `--pattern`：checkpoint 文件匹配规则，默认 `checkpoint_*.pkl`。
- `--start_epoch` / `--end_epoch`：限制评估 epoch 范围。
- `--limit`：最多评估多少个 checkpoint。
- `--resume`：跳过排名 JSON 中已经成功评估的 checkpoint。
- `--copy_best`：复制最佳权重为 `best_checkpoint.pkl`。
- `--data_config`：显式指定用于验证的数据配置。
- `--use_cuda 0`：使用 CPU 验证；默认使用 CUDA。

例如只评估 epoch 80 至 99，并支持断点继续：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --start_epoch 80 \
  --end_epoch 99 \
  --resume \
  --copy_best
```


## 固定 local_test 评测

`precompute_clean_points.py` 生成的固定 `local_test` 可用于公平比较不同 checkpoint。
`evaluate_local_test_models.py` 使用完整点云计算与比赛一致的逐样本 CD/P2S 百分制分数，
每个模型完成后立即把结果追加到 `result.txt`；再次执行会跳过已经记录的模型。

```bash
python scripts/evaluate_local_test_models.py \
  --model "VM L1 2.0" vm \
    checkpoint_selection_L1_2.0/best_checkpoint.pkl \
  --model "StraightPCF L1 2.0" straightpcf \
    checkpoint_selection_straightpcf_L1_2.0_1/best_checkpoint.pkl \
  --fusion-mode max
```

默认输入结构：

```text
dataset_train_pcd_disk/local_test/shapenet/<synset>/<model>/
├── clean.npy
├── noisy.npy
└── normalization.npz
```

P2S 使用对应的：

```text
dataset_train/local_test/shapenet/<synset>/<model>/models/model_normalized.obj
```

`result.txt` 是本地实验结果，不纳入 Git。

## HTML 点云可视化

`visualize_local_test_predictions.py` 对 local_test 每个 ShapeNet 类别选择模型 ID 排序后的
第一个样本，使用指定 checkpoint 推理，并生成可交互 HTML。Clean 为绿色、Prediction 为
蓝色、Noisy 为红色；页面按钮和图例可以分别开关三组点云。

VM 示例：

```bash
python scripts/visualize_local_test_predictions.py \
  checkpoint_selection_L1_2.0/best_checkpoint.pkl
```

StraightPCF 示例：

```bash
python scripts/visualize_local_test_predictions.py \
  checkpoint_selection_straightpcf_L1_2.0_1/best_checkpoint.pkl \
  --model-config straightpcf
```

默认输出：

```text
visualizations/<checkpoint目录名>/index.html
```

默认每组最多显示 50000 点。如果浏览器较慢，可以只降低显示点数；模型推理仍使用完整输入：

```bash
python scripts/visualize_local_test_predictions.py \
  checkpoint_selection_straightpcf_L1_2.0_1/best_checkpoint.pkl \
  --model-config straightpcf \
  --max-display-points 15000
```

`visualizations/` 是本地可视化产物，不纳入 Git，也不会保存额外的 `denoised.npy`。

## 推理

修改 `configs/task/predict_vm.yaml` 中的权重路径：

```yaml
load_ckpt: checkpoint_selection/best_checkpoint.pkl
```

然后运行：

```bash
python run.py --task configs/task/predict_vm.yaml
```

预测配置使用独立的空 `predict_transform`，不会对已经含噪的 `noisy.npy` 再次添加噪声。结果保存到：

```text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
```

每个输出应满足：

```text
shape 与输入 noisy.npy 完全相同
dtype 为 np.float32
```

## 验证预测输出

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

noisy_root = Path('dataset_test_noisy')
result_root = Path('results/dataset_test_noisy')
errors = []

for noisy_path in noisy_root.glob('shapenet/*/*/noisy.npy'):
    relative = noisy_path.relative_to(noisy_root)
    output_path = result_root / relative.parent / 'denoised.npy'
    if not output_path.exists():
        errors.append(f'缺少输出: {output_path}')
        continue

    noisy = np.load(noisy_path, mmap_mode='r')
    denoised = np.load(output_path, mmap_mode='r')
    if denoised.shape != noisy.shape:
        errors.append(f'shape 错误: {output_path}: {denoised.shape} != {noisy.shape}')
    if denoised.dtype != np.float32:
        errors.append(f'dtype 错误: {output_path}: {denoised.dtype}')
    if not np.isfinite(denoised).all():
        errors.append(f'包含 NaN/Inf: {output_path}')

if errors:
    print('\n'.join(errors))
    raise SystemExit(f'验证失败，共 {len(errors)} 个问题')
print('验证通过：所有 denoised.npy 的 shape、dtype 和数值均正常')
PY
```

## 打包提交

```bash
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

最终压缩包结构必须是：

```text
result.zip
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── denoised.npy
```


## Jittor StraightPCF（CVM + DistanceModule）


当前提交保留本地最佳结果所对应的基础设置：

```yaml
finetune_weight: 1.0
seed_k: 6
fusion_mode: best
```

对应最佳权重保存位置为
`checkpoint_selection_straightpcf_L1_2.0_1/best_checkpoint.pkl`。Checkpoint 文件本身体积较大且
已被 `.gitignore` 排除，需要在运行环境中自行保留或复制到该路径。

本分支补齐了 StraightPCF 的后两个训练阶段。完整训练顺序不可交换：

1. 训练单个 VelocityModule（已有 baseline）。
2. 将同一个第一阶段 VM 最优权重复制初始化多个 VelocityModule，联合训练 Coupled VelocityModule（CVM）。
3. 加载训练完成的 CVM，冻结其参数，训练 DistanceModule 和最终位置损失。

实现仍使用 Jittor，输入和输出点数完全相同。正式训练使用缓存 clean point cloud，但噪声、patch 和时间步仍在每次取样时动态生成。

### 第一阶段：准备 VelocityModule 最优权重

默认 CVM 配置从最新的缓存版 Charbonnier VM 最优权重初始化四个 VelocityModule：

~~~text
checkpoint_selection_L1_2.0/best_checkpoint.pkl
~~~

对应配置位于 configs/model/cvm.yaml：

~~~yaml
init_velocity_ckpt: checkpoint_selection_L1_2.0/best_checkpoint.pkl
num_modules: 4
~~~

如果 VM 最优权重位于其他目录，请先修改 init_velocity_ckpt。

如需重新训练 baseline：

~~~bash
python run.py --task configs/task/train_vm_cached.yaml
~~~

### 第二阶段：正式训练 Coupled VelocityModule

~~~bash
python run.py --task configs/task/train_cvm_cached.yaml
~~~

checkpoint 保存在：

~~~text
experiments/cvm_L1_2.0/checkpoint_<epoch>.pkl
~~~

使用默认的两阶段综合指标（loss:CD:P2S = 1:2:2）筛选 CVM：

~~~bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments/cvm_L1_2.0 \
  --task_template configs/task/train_cvm_cached.yaml \
  --mesh_dir dataset_train/local_train \
  --output_dir checkpoint_selection_cvm_L1_2.0 \
  --prefilter_top_k 10 \
  --copy_best
~~~

筛选结果为：

~~~text
checkpoint_selection_cvm_L1_2.0/best_checkpoint.pkl
~~~

### 第三阶段：正式训练 DistanceModule

训练前修改 configs/model/straightpcf.yaml：

~~~yaml
init_cvm_ckpt: checkpoint_selection_cvm_L1_2.0/best_checkpoint.pkl
~~~

然后运行：

~~~bash
python run.py --task configs/task/train_straightpcf_cached.yaml
~~~

此阶段会冻结 CVM 参数，只训练 DistanceModule。checkpoint 保存在：

~~~text
experiments/straightpcf_L1_2.0_1/checkpoint_<epoch>.pkl
~~~

筛选完整 StraightPCF checkpoint：

~~~bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments/straightpcf_L1_2.0_1 \
  --task_template configs/task/train_straightpcf_cached.yaml \
  --mesh_dir dataset_train/local_train \
  --output_dir checkpoint_selection_straightpcf_L1_2.0_1 \
  --prefilter_top_k 10 \
  --copy_best
~~~

### StraightPCF 预测

预测前需要确认两个路径。

configs/model/straightpcf.yaml：

~~~yaml
init_cvm_ckpt: checkpoint_selection_cvm_L1_2.0/best_checkpoint.pkl
~~~

configs/task/predict_straightpcf.yaml：

~~~yaml
load_ckpt: checkpoint_selection_straightpcf_L1_2.0_1/best_checkpoint.pkl
~~~

运行：

~~~bash
python run.py --task configs/task/predict_straightpcf.yaml
~~~

预测仍使用 configs/transform/predict.yaml 的空 predict_transform，不会给测试集 noisy.npy 二次加噪。输出目录和 baseline 相同：

~~~text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
~~~

输出验证和打包命令与前文 baseline 完全相同。
