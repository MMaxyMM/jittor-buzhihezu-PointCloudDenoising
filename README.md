# Jittor StraightPCF 点云降噪

本项目面向基于 ShapeNet 的点云降噪比赛：输入 noisy.npy，模型预测每个
输入点的三维位移，输出同点数、同顺序的 denoised.npy。最终保留方案对应：

~~~text
checkpoint_selection_straightpcf_maxagg_endpoint/best_checkpoint.pkl
~~~

该模型由 shared-patch-t VelocityModule、max-aggregation Coupled
VelocityModule 和 endpoint 联合微调 StraightPCF 三阶段训练得到。正式
预测使用 best patch 融合和 residual_alpha=1.10。

数据集、模型权重、日志和预测结果不提交到 Git；本文给出从官方 OBJ 数据
重新生成缓存、训练、筛选、评测和预测的完整流程。

## 1. 最终方法

1. **干净点云缓存**：每个训练 OBJ 预采样 200000 个表面点，并保存全部
   原始顶点。训练时动态抽取 32768 点并重新加噪，不固定 noisy 或 patch。
2. **Laplace + Charbonnier**：噪声尺度在 0.005～0.020 均匀采样，配置值
   直接作为 numpy.random.laplace 的 scale；三个阶段均使用平滑 L1。
3. **patch 共享时间**：同一个 patch 内所有点使用相同插值时间 t，保持
   Dynamic KNN 观察到的局部几何一致。
4. **EdgeConv max aggregation**：CVM 和最终 StraightPCF 使用 max 聚合。
5. **endpoint 联合微调**：StraightPCF 阶段保持 velocity 网络为 eval
   模式以固定 BatchNorm/Dropout 状态，但 endpoint loss 仍可更新其参数。
6. **最终推理**：每点采用覆盖它的最高权重 patch 预测，最后计算
   pred = noisy + 1.10 × (pred - noisy)。

configs/transform/predict.yaml 的 predict_transform 为空，不会对官方
noisy.npy 再次加噪。

## 2. 代码结构

~~~text
.
├── run.py
├── select_best_checkpoint.py
├── evaluate.py
├── requirements.txt
├── LICENSE
├── NOTICE
├── configs/
│   ├── data/
│   ├── model/
│   │   ├── vm.yaml
│   │   ├── cvm_maxagg_shared_patch_t.yaml
│   │   └── straightpcf_maxagg_endpoint.yaml
│   ├── system/
│   ├── task/
│   │   ├── train_vm_shared_patch_t.yaml
│   │   ├── train_cvm_maxagg_shared_patch_t.yaml
│   │   ├── train_straightpcf_maxagg_endpoint.yaml
│   │   └── predict_straightpcf_maxagg_endpoint.yaml
│   └── transform/
├── datalist/
├── scripts/
│   ├── create_local_holdout.py
│   ├── precompute_clean_points.py
│   ├── generate_local_test_benchmark.py
│   ├── estimate_noise_level.py
│   ├── evaluate_local_test_models.py
│   └── visualize_local_test_predictions.py
└── src/
    ├── data/
    ├── model/
    └── system/
~~~

- run.py：统一 train/predict 入口；--seed 同时设置 Jittor、NumPy 和 Python。
- src/model/feature.py：动态图 KNN 和 EdgeConv mean/max 聚合。
- src/model/vm.py：VelocityModule 与大点云 patch 推理。
- src/model/straightpcf.py：CVM、DistanceModule 和 endpoint 联合训练。
- src/data/augment.py：采样、归一化、动态噪声、线性增强和 patch 构造。
- select_best_checkpoint.py：loss/CD/P2S 两阶段 checkpoint 筛选。
- scripts/precompute_clean_points.py：训练缓存、固定 local_test 和噪声估计。
- scripts/evaluate_local_test_models.py：固定 local_test 的 CD/P2S 评测。
- scripts/visualize_local_test_predictions.py：交互式 HTML 点云对比。

## 3. 环境安装

### 3.1 审查目标环境

~~~text
Ubuntu 22.04
NVIDIA RTX 4090
CUDA 12.4
Python 3.10
Jittor >= 1.3.10
~~~

项目不依赖 PyTorch 或 JittorGeometric。第一次运行 Jittor 会编译 CUDA
算子，通常明显慢于后续运行。

~~~bash
conda create -n jittor_pcd python=3.10 -y
conda activate jittor_pcd
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
~~~

requirements.txt 包含带兼容范围的 Jittor、NumPy、SciPy、trimesh、
OmegaConf、point-cloud-utils、Plotly 和 tqdm。

~~~bash
nvidia-smi
nvcc --version
python -c "import jittor as jt; jt.flags.use_cuda=1; print((jt.ones((8,8))*2).mean().numpy())"
~~~

若 Jittor 没有找到 CUDA，请检查驱动、CUDA 12.4 工具链和
LD_LIBRARY_PATH，并参考 Jittor 官方安装文档。

### 3.2 CPU 线程建议

16 个 DataLoader worker 下建议在训练终端执行：

~~~bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
~~~

这可避免每个 worker 再创建多组 BLAS/OpenMP 线程。本仓库不上传 Bash
辅助脚本，直接设置环境变量后运行原 Python 命令即可。

## 4. 数据准备

### 4.1 官方目录

~~~text
dataset_train/
└── shapenet/<synset_id>/<model_id>/models/model_normalized.obj

dataset_test_noisy/
└── shapenet/<synset_id>/<model_id>/noisy.npy
~~~

noisy.npy 应为有限数值的 (N, 3) float32。datalist/train.txt、
datalist/validate.txt 和 datalist/test.txt 每行格式为：

~~~text
shapenet/<synset_id>/<model_id>
~~~

如数据不在项目根目录，修改 configs/data/train_cached.yaml 和
configs/data/predict.yaml 的 input_dataset_dir。

### 4.2 创建固定留出集

按 ShapeNet 类别确定性留出 2%，且每类至少 2 个模型：

~~~bash
python scripts/create_local_holdout.py \
  --dataset_dir dataset_train \
  --test_ratio 0.02 \
  --min_test_per_category 2 \
  --seed 123
~~~

脚本只创建相对目录软链接，不移动或复制 OBJ：

~~~text
dataset_train/
├── shapenet/
├── local_train/
│   ├── datalist.txt
│   └── shapenet/<synset>/<model> -> 原始模型目录
├── local_test/
│   ├── datalist.txt
│   └── shapenet/<synset>/<model> -> 原始模型目录
└── local_split_manifest.json
~~~

相对链接位于持久磁盘时，服务器重启后仍存在。移动原始 shapenet 或删除
链接目标才会失效。重新划分使用 --overwrite；它不会删除原始 OBJ。

### 4.3 生成缓存和固定 local_test

~~~bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd_disk \
  --train_num_points 200000 \
  --test_num_points 50000 \
  --num_vertex_samples 1024 \
  --workers 16 \
  --seed 123
~~~

~~~text
dataset_train_pcd_disk/
├── local_train/shapenet/<synset>/<model>/
│   ├── clean.npy
│   └── vertices.npy
└── local_test/shapenet/<synset>/<model>/
    ├── clean.npy
    ├── noisy.npy
    └── normalization.npz
~~~

local_train 中 clean.npy 是 200000×3 float32 表面点池，vertices.npy
保存全部 OBJ 顶点。每次训练随机保留最多 1024 个原始顶点，再从表面池
补齐 32768 点，然后重新归一化、动态加噪和构造 patch。

local_test 固定生成 50000 点 clean/noisy，并标定噪声使 noisy 基线尽量
接近 mean_CD=0.000246、mean_P2S=0.000196。normalization.npz 保存
P2S 所需的 center 和 scale。

只生成训练缓存：

~~~bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd_disk \
  --splits local_train \
  --workers 16 \
  --seed 123
~~~

小规模检查应使用独立目录：

~~~bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir /tmp/dataset_train_pcd_smoke \
  --limit 1 \
  --workers 1 \
  --noise_scale 1.0
~~~

脚本默认跳过完整缓存。改变划分、点数或噪声标定后，应使用 --overwrite
统一重建，避免混合不同参数。

## 5. 最终配置

| 阶段 | task 配置 | epoch | 关键参数 |
|---|---|---:|---|
| VM | train_vm_shared_patch_t.yaml | 100 | lr=1e-4，patch=1000，监督点=128，共享 patch t |
| CVM | train_cvm_maxagg_shared_patch_t.yaml | 80 | 4 个 VM，EdgeConv max，iterations=3，consistency=10 |
| StraightPCF | train_straightpcf_maxagg_endpoint.yaml | 50 | EdgeConv max，iterations=2，finetune_weight=1 |
| 推理 | predict_straightpcf_maxagg_endpoint.yaml | - | best patch，seed_k=6，alpha=1.10 |

训练缓存配置为 batch_size=16、num_workers=16、每个 epoch 随机使用
10000 个模型条目；验证 batch_size=1、num_workers=8。

## 6. 三阶段训练和筛选

三个阶段顺序不可交换。以下命令统一使用 seed=123。

### 6.1 VelocityModule

~~~bash
python run.py \
  --task configs/task/train_vm_shared_patch_t.yaml \
  --seed 123
~~~

checkpoint 位于 experiments/vm_shared_patch_t/。筛选最后 70 个，快速 CD
初筛 10 个后完整计算 loss/CD/P2S：

~~~bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments/vm_shared_patch_t \
  --task_template configs/task/train_vm_shared_patch_t.yaml \
  --mesh_dir dataset_train/local_train \
  --output_dir checkpoint_selection_vm_shared_patch_t \
  --loss_weight 1 --cd_weight 2 --p2s_weight 2 \
  --cd_limit 20 \
  --prefilter_top_k 10 \
  --prefilter_cd_points 4096 \
  --prefilter_cd_limit 10 \
  --prefilter_last_n 70 \
  --validation_workers 0 \
  --seed 123 \
  --copy_best
~~~

下一阶段需要：

~~~text
checkpoint_selection_vm_shared_patch_t/best_checkpoint.pkl
~~~

### 6.2 Coupled VelocityModule

~~~bash
python run.py \
  --task configs/task/train_cvm_maxagg_shared_patch_t.yaml \
  --seed 123
~~~

~~~bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments/cvm_maxagg_shared_patch_t \
  --task_template configs/task/train_cvm_maxagg_shared_patch_t.yaml \
  --mesh_dir dataset_train/local_train \
  --output_dir checkpoint_selection_cvm_maxagg_shared_patch_t \
  --loss_weight 1 --cd_weight 2 --p2s_weight 2 \
  --cd_limit 20 \
  --prefilter_top_k 5 \
  --prefilter_cd_points 4096 \
  --prefilter_cd_limit 10 \
  --prefilter_last_n 30 \
  --validation_workers 0 \
  --seed 123 \
  --copy_best
~~~

下一阶段和最终模型初始化需要：

~~~text
checkpoint_selection_cvm_maxagg_shared_patch_t/best_checkpoint.pkl
~~~

### 6.3 StraightPCF endpoint 联合微调

~~~bash
python run.py \
  --task configs/task/train_straightpcf_maxagg_endpoint.yaml \
  --seed 123
~~~

~~~bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments/straightpcf_maxagg_endpoint \
  --task_template configs/task/train_straightpcf_maxagg_endpoint.yaml \
  --mesh_dir dataset_train/local_train \
  --output_dir checkpoint_selection_straightpcf_maxagg_endpoint \
  --loss_weight 1 --cd_weight 2 --p2s_weight 2 \
  --cd_limit 20 \
  --prefilter_top_k 5 \
  --prefilter_cd_points 4096 \
  --prefilter_cd_limit 10 \
  --prefilter_last_n 30 \
  --validation_workers 0 \
  --seed 123 \
  --copy_best
~~~

最终权重：

~~~text
checkpoint_selection_straightpcf_maxagg_endpoint/best_checkpoint.pkl
~~~

本次最佳模型来自 epoch 38。composite 使用各指标的加权排名而非直接相加
不同量纲的值，权重为 loss:CD:P2S=1:2:2。评估失败或包含 NaN/Inf 的
checkpoint 不会被选为最佳模型。

## 7. 正式推理

必须准备：

~~~text
checkpoint_selection_cvm_maxagg_shared_patch_t/best_checkpoint.pkl
checkpoint_selection_straightpcf_maxagg_endpoint/best_checkpoint.pkl
dataset_test_noisy/shapenet/<synset>/<model>/noisy.npy
datalist/test.txt
~~~

最终权重包含完整参数，但模型构造阶段仍会读取配置中的 CVM 初始化权重。

~~~bash
python run.py \
  --task configs/task/predict_straightpcf_maxagg_endpoint.yaml \
  --seed 123
~~~

输出：

~~~text
results/dataset_test_noisy/shapenet/<synset>/<model>/denoised.npy
~~~

denoised.npy 必须为 (N, 3) float32，N 与 noisy.npy 完全一致。可使用以下
Python 命令检查：

~~~bash
python -c "from pathlib import Path; import numpy as np; a=list(Path('dataset_test_noisy').glob('shapenet/*/*/noisy.npy')); b=list(Path('results/dataset_test_noisy').glob('shapenet/*/*/denoised.npy')); assert len(a)==len(b); print('files:',len(b))"
~~~

打包时进入 results/dataset_test_noisy 后执行：

~~~bash
zip -r ../../result.zip shapenet/
~~~

zip 内第一层必须直接是 shapenet/。

## 8. 固定 local_test 评测

该脚本不保存 denoised.npy，而是逐样本推理后直接计算指标，并在每个模型
完成后追加结果。重复运行会跳过已记录的模型来源。

~~~bash
python scripts/evaluate_local_test_models.py \
  --model "StraightPCF maxagg endpoint-grad alpha1.10" \
    configs/model/straightpcf_maxagg_endpoint.yaml \
    checkpoint_selection_straightpcf_maxagg_endpoint/best_checkpoint.pkl \
  --data-root dataset_train_pcd_disk/local_test \
  --mesh-root dataset_train/local_test \
  --datalist dataset_train/local_test/datalist.txt \
  --fusion-mode max \
  --result-file outputs/local_test/maxagg_endpoint/result.txt \
  --seed 123
~~~

- CD：在 clean 点云决定的单位球坐标中计算双向平均最近邻平方距离之和。
- P2S：使用 point-cloud-utils 计算预测点到 OBJ 三角面的最近距离平方均值。
- 百分制：逐样本以 noisy 为零分基线，CD/P2S 各占 50%。

当前固定 local_test 记录：

~~~text
score: 76.5039
mean_CD: 0.00008848
mean_P2S: 0.00001964
~~~

这是本地代理结果，不是官方隐藏测试集成绩。留出样本、缓存采样和噪声标定
都会造成与线上分数的差异。

## 9. HTML 可视化

每个 ShapeNet 类别选择第一个模型，输出 clean 绿色、prediction 蓝色、
noisy 红色的交互 HTML：

~~~bash
python scripts/visualize_local_test_predictions.py \
  checkpoint_selection_straightpcf_maxagg_endpoint/best_checkpoint.pkl \
  --model-config configs/model/straightpcf_maxagg_endpoint.yaml \
  --data-root dataset_train_pcd_disk/local_test \
  --datalist dataset_train/local_test/datalist.txt \
  --output-dir outputs/visualizations/maxagg_endpoint \
  --max-display-points 15000
~~~

max-display-points 只减少 HTML 显示点数，推理仍使用完整输入。

## 10. 复现注意事项

- 所有入口均提供 seed，本文统一使用 123；多进程、GPU KNN 和 JIT 编译在
  不同硬件上仍可能带来微小数值差异。
- OBJ 的 material not found in .mtl 警告只涉及材质。模型仅使用顶点和
  三角面，颜色、纹理和材质不参与训练。
- predict_transform 必须保持为空，禁止给官方 noisy.npy 二次加噪。
- 数据、权重、npy/npz、日志、HTML、zip、outputs 和 Bash 脚本均由
  .gitignore 排除。
- 建议将实际命令、配置快照和终端日志保存在 outputs/，该目录不进入 Git。

## 11. 第三方来源

本项目基于比赛 Jittor starter code，并将 StraightPCF 三阶段架构移植到
Jittor。第三方说明见 NOTICE。

StraightPCF 官方代码：

https://github.com/ddsediri/StraightPCF

~~~bibtex
@InProceedings{de_Silva_Edirimuni_2024_CVPR,
  author    = {de Silva Edirimuni, Dasith and Lu, Xuequan and Li, Gang
               and Wei, Lei and Robles-Kelly, Antonio and Li, Hongdong},
  title     = {StraightPCF: Straight Point Cloud Filtering},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition},
  pages     = {20721--20730},
  year      = {2024}
}
~~~

仓库代码按 LICENSE 发布；第三方依赖和 StraightPCF 原实现仍适用各自许可。
