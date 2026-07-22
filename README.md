# 点云降噪（Point Cloud Denoising）Baseline

基于 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 的点云降噪 Baseline。给定含噪点云，预测逐点位移，将点“推回”真实表面附近，输出与输入点数相同的降噪点云。

方法参考：[StraightPCF](https://openaccess.thecvf.com/content/CVPR2024/html/Edirimuni_StraightPCF_Straight_Point_Cloud_Filtering_CVPR_2024_paper.html)（CVPR 2024）。本仓库在官方 starter 基础上保留 OBJ 训练流程，并提供 clean 点云缓存训练、拉普拉斯噪声适配，以及完整 StraightPCF 三阶段训练。

---

## 1. 赛题简介

三维传感（LiDAR、结构光、深度相机等）获取的点云常受传感器误差、环境干扰与量化误差影响，点偏离真实表面，损害重建、法向估计、识别等下游任务。

**任务**：输入含噪点 p_i + n_i_{i=1}^{N}，预测位移 \Delta_i，使降噪点 \hat{p}_i = p_i + n_i + \Delta_i 尽可能贴近真实表面。可建模为逐点位移回归，或分数匹配 / 扩散式去噪。

**主要挑战**：噪声分布多样、尖锐细节保持、大规模点云可扩展性、跨类别泛化。

---



## 2. 数据说明


| 资源                      | 文件                       |
| ----------------------- | ------------------------ |
| A 榜训练集（干净网格）            | `dataset_train.tar.gz`   |
| A 榜测试集（含噪点云）            | `dataset_test_noisy.zip` |
| Baseline / Starter Code | `starter_code.zip`       |


- **A 榜**：中等规模 ShapeNet 子集（飞机、椅子、桌子、汽车等），每样本约 50,000 点，噪声标准差 0.005～0.020（单位球归一化后）。
- **B 榜**：更大规模、更多类别与更高噪声，需兼顾效率；AB 榜算法应基本一致。
- **禁止**使用提供数据集以外的数据，否则取消资格。



### 目录结构

**训练集**（约 2 万模型，`.obj` 干净网格；自行采样加点噪，Baseline 已含流程）：

```text
dataset_train/
  shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

**测试集**（仅含噪 `.npy`，`float32`，shape `(N, 3)`；GT 由组委会持有）：

```text
dataset_test_noisy/
  shapenet/<synset_id>/<model_id>/noisy.npy
```

`datalist/train.txt`、`datalist/validate.txt`、`datalist/test.txt` 中为相对数据集根目录的模型路径，例如：

```text
shapenet/04401088/d7ed512f7a7daf63772afc88105fa679
```



### 数据准备

```bash
tar xzf dataset_train.tar.gz
unzip dataset_test_noisy.zip
```

---



## 3. Baseline 架构


| 模块   | 说明                                       |
| ---- | ---------------------------------------- |
| 特征提取 | 动态图卷积（Dynamic Edge Convolution）+ KNN 局部图 |
| 位移解码 | MLP → 三维位移向量                             |
| 训练目标 | 监督学习：含噪点相对干净点的位移回归                       |
| 推理   | Patch 分块降噪 + 加权融合，适配大规模点云                |




### 本仓库相对官方 starter 的适配

1. **噪声建模**（`src/data/augment.py`）：`noise_type` 支持 `laplace`（默认）与 `gaussian`。配置中 `noise_std_min/max` 表示噪声标准差；拉普拉斯采样时换算为尺度 b = \mathrm{std}/\sqrt{2}。
2. **损失函数**（`src/model/vm.py`、`src/model/straightpcf.py`）：三阶段统一使用 Charbonnier（平滑 L1）损失 \sqrt{d^2 + \varepsilon}，对重尾噪声更鲁棒。
3. **推理融合**（`src/model/vm.py`）：多 patch 按 \exp(-\mathrm{dist}) 加权融合，并用 scatter 向量化加速。

---



## 4. 环境安装

推荐 Python 3.9，GCC/G++ ≤ 10。本赛题**必须**使用 Jittor。

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
```

`requirements.txt` 含：`jittor`、`numpy`、`trimesh`、`scipy`、`omegaconf`。

本地评测需精确 P2S 时额外安装：

```bash
pip install point-cloud-utils
```

文档：[Jittor Docs](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)。环境要求：Ubuntu ≥ 16.04 或 WSL；Python ≥ 3.7；g++ ≥ 5.4。Windows 需通过 WSL（当前 WSL 尚不支持 CUDA）。

也可通过 Docker / pip / 源码安装 Jittor，或设置 `cc_path` / `nvcc_path` 指定编译器。启用 CUDA 示例：

```bash
export nvcc_path="/usr/local/cuda/bin/nvcc"
python3 -m jittor.test.test_cuda
```



### 多 worker 训练时限制 CPU 线程

DataLoader `num_workers` 较大时，NumPy/BLAS 可能造成 CPU 过度订阅。训练前在当前终端执行：

```bash
source scripts/run_single_thread.sh
```

会设置 `OMP_NUM_THREADS` 等为 1（仅当前终端有效，需用 `source`）。

---



## 5. 训练



### 5.1 原始 OBJ 训练

每次读取 OBJ，动态做表面采样、归一化、加噪与 patch 构造：

```bash
python run.py --task configs/task/train_vm.yaml
```

权重默认：`experiments/vm/checkpoint_<epoch>.pkl`。

### 5.2 Clean 点云缓存（推荐）

预采样可减少每 epoch 解析 OBJ / mesh 采样的 CPU/IO 开销。`scripts/precompute_clean_points.py` 输出：

```text
dataset_train_pcd/
  shapenet/<synset_id>/<model_id>/
    clean.npy      # 默认 200000 个按面积采样的表面点
    vertices.npy   # OBJ 全部原始顶点
```

训练时随机取最多 1024 个原始顶点，再从表面点池补齐到 32768 点；随后仍动态加噪并构造 patch，不会固定 noisy 数据。

```bash
# 小规模测试
python scripts/precompute_clean_points.py \
  --input_dir dataset_train --output_dir dataset_train_pcd \
  --num_points 200000 --workers 8 --seed 123 --limit 100

# 完整缓存（约 15833 个模型，200000 点 float32 约 38 GB）
python scripts/precompute_clean_points.py \
  --input_dir dataset_train --output_dir dataset_train_pcd \
  --num_points 200000 --workers 16 --seed 123
```

已有文件默认跳过，可断点续跑；升级旧版点数或强制重生成需加 `--overwrite`。`/dev/shm` 空间充足时可写到内存盘再软链到 `dataset_train_pcd`。

数量检查：

```bash
find dataset_train -path '*/models/model_normalized.obj' -type f | wc -l
find dataset_train_pcd -name clean.npy -type f | wc -l
find dataset_train_pcd -name vertices.npy -type f | wc -l
```

缓存训练（每 epoch 10000 个训练样本）：

```bash
python run.py --task configs/task/train_vm_cached.yaml
```

数据流：`vertices.npy` + `clean.npy` → 抽样至 32768 点 → 归一化 → 动态 Laplace 噪声 → 1000 点局部 patch → 训练位移 / velocity target。

### 5.3 StraightPCF 三阶段（顺序不可交换）

1. 训练 VelocityModule（VM）
2. 用同一 VM 最优权重复制初始化多个模块，联合训练 Coupled VelocityModule（CVM）
3. 加载并冻结 CVM，训练 DistanceModule

```bash
# 阶段 1：VM（若尚未训练）
python run.py --task configs/task/train_vm_cached.yaml

# 阶段 2：确认 configs/model/cvm.yaml 中
#   init_velocity_ckpt: checkpoint_selection_cached/best_checkpoint.pkl
python run.py --task configs/task/train_cvm_cached.yaml

# 阶段 3：确认 configs/model/straightpcf.yaml 中
#   init_cvm_ckpt: checkpoint_selection_cvm/best_checkpoint.pkl
python run.py --task configs/task/train_straightpcf_cached.yaml
```

对应 checkpoint 目录：`experiments/vm`、`experiments/cvm`、`experiments/straightpcf`。

### 5.4 条件残差扩散与消融模型

扩散分支保持训练数据、patch 划分和提交格式不变，在归一化残差
`r0 = (clean - noisy) / noise_std` 上训练条件扩散模型。默认使用
DGCNN-like 编码器、v-prediction、`0.1 × L1` 辅助项和 4 步确定性
DDIM；测试时默认噪声条件为 0.0125，也可在模型配置中将
`inference_noise_mode` 改为 `estimate`。

```bash
# 同 backbone 的直接端点残差基线
python run.py --task configs/task/train_direct_residual_cached.yaml

# DGCNN 条件残差扩散（首选实验）
python run.py --task configs/task/train_residual_diffusion_cached.yaml

# 仅在 DGCNN 扩散通过验证门槛后比较局部 Point Transformer
python run.py --task configs/task/train_direct_residual_point_transformer_cached.yaml
python run.py --task configs/task/train_residual_diffusion_point_transformer_cached.yaml
```

在正式训练前可用一个真实缓存样本完成 GPU 前向、反向和优化器冒烟测试：

```bash
python scripts/smoke_diffusion_pipeline.py --use_cuda 1
```

模型配置中的 `num_inference_steps` 可用于比较 1/2/4/8/16 步；
`prediction_type` 支持 `v` 与 `epsilon`；`condition_on_time` 和
`condition_on_observation_std` 可关闭以完成条件消融。训练系统还支持
可选的 `adamw`、`gradient_clip_norm`、`ema_decay` 以及
`scheduler: cosine` + `warmup_ratio`，默认配置仍使用 Adam，以免混淆
扩散机制与优化器收益。

`run.py` 与 `select_best_checkpoint.py` 均支持重复传入
`--model_override key=value`，例如使用同一权重比较 8 步采样：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/residual_diffusion \
  --task_template configs/task/train_residual_diffusion_cached.yaml \
  --metric cd --model_override num_inference_steps=8
```

---



## 6. 选择最佳 Checkpoint

`select_best_checkpoint.py` 在本地验证集上评估，不需要官方测试 GT。可用 validation loss，或 `--metric cd`（动态加噪后算 Chamfer Distance，与竞赛更对齐）。

```bash
# 按 validation loss（缓存 VM 示例）
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --task_template configs/task/train_vm_cached.yaml \
  --output_dir checkpoint_selection_cached \
  --copy_best

# 按 CD（噪声种子固定，默认 std 0.005~0.020）
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --task_template configs/task/train_vm_cached.yaml \
  --metric cd --cd_limit 50 \
  --output_dir checkpoint_selection_cd \
  --copy_best
```

CVM / StraightPCF 同理，改 `--ckpt_dir`、`--task_template`、`--output_dir` 即可。常用参数：`--pattern`、`--start_epoch` / `--end_epoch`、`--limit`、`--resume`、`--use_cuda 0`。

输出示例：

```text
checkpoint_selection_cached/
├── checkpoint_ranking.csv
├── checkpoint_ranking.json
├── best_checkpoint.pkl
└── logs/
```

辅助工具：估计测试集噪声水平

```bash
python scripts/estimate_noise_level.py --input_dir dataset_test_noisy
```

若测试噪声集中在某固定值，可将配置中 `noise_std_min/max` 收窄对准。模型配置可设 `predict_rounds`（多轮迭代降噪，默认 1；>1 需在验证集确认不过度收缩）。

---



## 7. 推理与提交



### 推理

在对应 predict 配置中设置权重路径，例如 `configs/task/predict_vm.yaml`：

```yaml
load_ckpt: checkpoint_selection_cached/best_checkpoint.pkl
```

StraightPCF 还需确认 `configs/model/straightpcf.yaml` 的 `init_cvm_ckpt`，以及 `configs/task/predict_straightpcf.yaml` 的 `load_ckpt`。

```bash
python run.py --task configs/task/predict_vm.yaml
# 或
python run.py --task configs/task/predict_straightpcf.yaml
```

预测使用空 `predict_transform`，不会对已含噪的 `noisy.npy` 再次加噪。结果：

```text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
```

要求：`shape` 与输入一致，`dtype` 为 `np.float32`。

### 验证输出

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



### 打包提交

```text
result.zip
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── denoised.npy   # float32, shape (N, 3)
```

```python
import numpy as np
np.save("denoised.npy", denoised_points.astype(np.float32))
```

每队每天最多提交 **2** 次。成绩有效须按赛事开源指南开源代码。

---



## 8. 评测指标

计算前将真实点云归一化至单位球（中心化 + 最大半径为 1），预测点施加相同变换。

### Chamfer Distance (CD)

预测点云与真实干净点云之间的双向平均最近邻**平方距离**之和，越小越好；本项目的 `evaluate.py` 与 checkpoint CD 筛选均按此口径计算。

\[
\mathrm{CD}(S_{\mathrm{pred}}, S_{\mathrm{gt}})
= \frac{1}{|S_{\mathrm{pred}}|}\sum_{x\in S_{\mathrm{pred}}}\min_{y\in S_{\mathrm{gt}}}\|x-y\|_2^2
+ \frac{1}{|S_{\mathrm{gt}}|}\sum_{y\in S_{\mathrm{gt}}}\min_{x\in S_{\mathrm{pred}}}\|y-x\|_2^2
\]

### Point-to-Surface (P2S)

预测点到原始干净网格表面的最近距离平方均值，越小越好；对过度平滑或表面偏移更敏感。

\[
\mathrm{P2S}(S_{\mathrm{pred}}, M)
= \frac{1}{|S_{\mathrm{pred}}|}\sum_{x\in S_{\mathrm{pred}}}\min_{y\in M}\|x-y\|_2^2
\]

### 百分制得分

以含噪输入为零分基线：

\[
\mathrm{cd\_score}_i = \mathrm{clamp}\left(100\times\left(1-\frac{\mathrm{CD}_{\mathrm{pred}}^{(i)}}{\mathrm{CD}_{\mathrm{noisy}}^{(i)}}\right),0,100\right)
\]

\[
\mathrm{p2s\_score}_i = \mathrm{clamp}\left(100\times\left(1-\frac{\mathrm{P2S}_{\mathrm{pred}}^{(i)}}{\mathrm{P2S}_{\mathrm{noisy}}^{(i)}}\right),0,100\right)
\]

**A 榜最终分**：全部测试样本上 CD 得分与 P2S 得分各取全局平均，再按 **50% / 50%** 加权。

**B 榜**：评测方式将据 A 榜结果调整；最终得分 = **65% 评测 + 35% 答辩**。A 榜结束后按排名筛选进入 B 榜。

### 本地评测（GT 通常仅组委会持有）

```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8 \
    --verbose
```

`--mesh_dir` 用于 P2S（需 `point-cloud-utils`）；省略则仅算 CD。

---



## 9. 注意事项

1. 需通过热身赛后方可参赛。
2. **必须**使用 Jittor；不得使用额外外部数据。
3. 不同类别可用不同网络 / 权重 / 超参。
4. 输出点数必须与输入含噪点云一致，否则评测报错。
5. 须按开源指南开源代码，成绩方有效。
6. 每队每天最多提交两次。
