from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .patch_inference import patch_based_denoise
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        self.patch_size = int(cfg.get('patch_size', 1000))
        self.seed_k = int(cfg.get('seed_k', 6))
        self.seed_k_alpha = int(cfg.get('seed_k_alpha', 1))
        self.num_inference_steps = int(cfg.get('num_inference_steps', 4))
        
        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim']
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        
        # target
        grad_dir_t_target = pc_clean - pc_noisy
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore

        # 拉普拉斯噪声的 MLE 对应 L1 损失；使用 Charbonnier（平滑 L1）
        # 既保留对重尾离群噪声的鲁棒性，又避免 L1 在零点处不可导
        diff = pred_dir - grad_dir_t_target
        loss = (jt.sqrt((diff ** 2.0).sum(dim=-1) + 1e-12) / self.dsm_sigma).mean()

        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        """
        pcl_noisy: (B, N, 3)
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        # predict_rounds: 将上一轮输出重新当作输入迭代降噪，
        # 对拉普拉斯重尾残留的离群点有效；>1 时需验证细节不过度收缩
        num_steps = int(self.model_config.get('predict_rounds', 1))
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(num_steps):
                pc_out = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.patch_size,
                    seed_k=self.seed_k,
                    seed_k_alpha=self.seed_k_alpha,
                    inner_steps=self.num_inference_steps,
                )
                if pc_out is None:
                    # patch 推理失败时回退上一步结果，保证输出完整
                    break
                pc_next = pc_out
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
