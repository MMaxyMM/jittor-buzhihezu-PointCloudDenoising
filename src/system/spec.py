from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import math
import os

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    parameters = (
        model.get_optim_parameters()
        if hasattr(model, "get_optim_parameters")
        else model.parameters()
    )
    optimizer = OptimizerClass(parameters, **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        self.scheduler = trainer_config.get('scheduler', None)
        self.warmup_ratio = float(trainer_config.get('warmup_ratio', 0.0))
        self.min_lr_ratio = float(trainer_config.get('min_lr_ratio', 0.0))
        self.gradient_clip_norm = trainer_config.get('gradient_clip_norm', None)
        self.ema_decay = float(trainer_config.get('ema_decay', 0.0))
        self.validate_every_n_epochs = int(
            trainer_config.get("validate_every_n_epochs", 1)
        )
        if self.validate_every_n_epochs < 1:
            raise ValueError("validate_every_n_epochs must be positive")
        self.checkpoint_every_n_epochs = int(
            trainer_config.get("checkpoint_every_n_epochs", 1)
        )
        if self.checkpoint_every_n_epochs < 1:
            raise ValueError("checkpoint_every_n_epochs must be positive")
        self.start_epoch = int(trainer_config.get("start_epoch", 0))
        self._global_step = int(trainer_config.get("global_step", 0))
        model_init_checkpoint = trainer_config.get("model_init_checkpoint")
        resume_checkpoint = trainer_config.get("resume_checkpoint")
        self.resume_checkpoint = resume_checkpoint
        if model_init_checkpoint and resume_checkpoint:
            raise ValueError(
                "set only one of model_init_checkpoint/resume_checkpoint"
            )
        initial_checkpoint = resume_checkpoint or model_init_checkpoint
        if initial_checkpoint:
            if not os.path.isfile(initial_checkpoint):
                raise FileNotFoundError(
                    f"initial checkpoint does not exist: {initial_checkpoint}"
                )
            model.load(initial_checkpoint)
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        self.base_lr = (
            float(self.optimizer.lr) if self.optimizer is not None else 0.0
        )
        if self.resume_checkpoint:
            state_path = f"{self.resume_checkpoint}.state.pkl"
            if os.path.isfile(state_path):
                state = jt.load(state_path)
                self.start_epoch = int(state.get("epoch", -1)) + 1
                self._global_step = int(
                    state.get("global_step", self._global_step)
                )
                optimizer_state = state.get("optimizer")
                if (
                    optimizer_state is not None
                    and self.optimizer is not None
                    and hasattr(self.optimizer, "load_state_dict")
                ):
                    self.optimizer.load_state_dict(optimizer_state)
        self._ema_parameters = None
        if self.ema_decay > 0.0:
            if not 0.0 < self.ema_decay < 1.0:
                raise ValueError("ema_decay must be in (0, 1)")
            parameters = (
                self.model.get_optim_parameters()
                if hasattr(self.model, "get_optim_parameters")
                else self.model.parameters()
            )
            self._ema_parameters = [
                parameter.detach().clone()
                for parameter in parameters
            ]
        
        self._validation_loss = defaultdict(list)
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def on_before_optimizer_step(self, optimizer):
        if self.gradient_clip_norm is not None:
            optimizer.clip_grad_norm(float(self.gradient_clip_norm))

    def _update_learning_rate(self, total_steps):
        if self.optimizer is None or self.scheduler is None:
            return
        progress = self._global_step / max(total_steps - 1, 1)
        if self.warmup_ratio > 0.0 and progress < self.warmup_ratio:
            factor = max(progress / self.warmup_ratio, 1.0 / max(total_steps, 1))
        elif self.scheduler == "cosine":
            post_warmup = (progress - self.warmup_ratio) / max(
                1.0 - self.warmup_ratio, 1e-8
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * post_warmup))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        else:
            raise ValueError(f"unsupported scheduler: {self.scheduler}")
        self.optimizer.lr = self.base_lr * factor

    def _update_ema(self):
        if self._ema_parameters is None:
            return
        with jt.no_grad():
            parameters = (
                self.model.get_optim_parameters()
                if hasattr(self.model, "get_optim_parameters")
                else self.model.parameters()
            )
            for ema_parameter, parameter in zip(
                self._ema_parameters, parameters
            ):
                ema_parameter.assign(
                    self.ema_decay * ema_parameter
                    + (1.0 - self.ema_decay) * parameter.detach()
                )

    def _save_checkpoint(self, checkpoint_path):
        if self._ema_parameters is None:
            self.model.save(checkpoint_path)
            return
        with jt.no_grad():
            parameters = list(
                self.model.get_optim_parameters()
                if hasattr(self.model, "get_optim_parameters")
                else self.model.parameters()
            )
            backup = [parameter.detach().clone() for parameter in parameters]
            for parameter, ema_parameter in zip(parameters, self._ema_parameters):
                parameter.assign(ema_parameter)
            self.model.save(checkpoint_path)
            for parameter, original in zip(parameters, backup):
                parameter.assign(original)

    def _save_training_state(self, checkpoint_path, epoch):
        state = {
            "epoch": int(epoch),
            "global_step": int(self._global_step),
            "optimizer": None,
        }
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            state["optimizer"] = self.optimizer.state_dict()
        jt.save(state, f"{checkpoint_path}.state.pkl")
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        train_dataloader = self.dataset_module.train_dataloader()
        assert train_dataloader is not None, "train_dataloader is None"
        steps_per_epoch = max(
            1, len(train_dataloader) // train_dataloader.batch_size
        )
        total_steps = self.epochs * steps_per_epoch
        # Reuse one dataloader for all epochs. Recreating workers every epoch
        # previously leaked shared memory and OOM-killed training on 16GB hosts.
        validate_dataloader = self.dataset_module.validate_dataloader()
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.on_train_epoch_start()
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            for batch in pbar:
                self.on_train_batch_start()
                self._update_learning_rate(total_steps)
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self._update_ema()
                self._global_step += 1
                self.on_train_batch_end()
            self.on_train_epoch_end()
            
            self.model.eval()
            should_validate = (
                (epoch + 1) % self.validate_every_n_epochs == 0
                or epoch + 1 == self.epochs
            )
            if validate_dataloader is not None and should_validate:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
            
            should_checkpoint = (
                (epoch + 1) % self.checkpoint_every_n_epochs == 0
                or epoch + 1 == self.epochs
            )
            if should_checkpoint:
                checkpoint_path = os.path.join(
                    self.ckpt_save_dir,
                    f'{self.ckpt_save_name}_{epoch}.pkl',
                )
                os.makedirs(self.ckpt_save_dir, exist_ok=True)
                self._save_checkpoint(checkpoint_path)
                self._save_training_state(checkpoint_path, epoch)
            if hasattr(jt, "gc"):
                jt.gc()
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")