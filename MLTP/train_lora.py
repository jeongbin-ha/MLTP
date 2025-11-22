"""
Diffusion 모델 LoRA 파인튜닝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel
from peft import LoraConfig, get_peft_model
import pickle

from config import LORA_CONFIG, DIFFUSION_CONFIG, DEVICE, SEED, CHECKPOINT_DIR
from utils import set_seed, Logger, AverageMeter


class LoRATrainer:
    """LoRA 파인튜닝 관리"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        lora_rank: int,
        config: dict = LORA_CONFIG,
        save_dir: Optional[Path] = None,
        device: str = DEVICE
    ):
        self.device = device
        self.config = config
        self.lora_rank = lora_rank
        
        # LoRA 설정
        alpha = int(lora_rank * config.get("alpha_scale", 16.0))  # alpha = rank * scale
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=alpha,
            lora_dropout=config["dropout"],
            target_modules=config["target_modules"],
            bias="none",
        )
        
        # LoRA 적용
        self.model = get_peft_model(model, lora_config)
        self.model.to(device)
        
        # 학습 가능한 파라미터만 출력
        self.model.print_trainable_parameters()
        print(f"LoRA rank: {lora_rank}, alpha: {alpha} (scale: {alpha/lora_rank:.1f})")
        
        self.train_loader = train_loader
        
        if save_dir is None:
            save_dir = CHECKPOINT_DIR / "lora"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Scheduler (linear warmup + cosine decay)
        num_training_steps = len(train_loader) * config["num_epochs"]
        num_warmup_steps = config["warmup_steps"]
        
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Noise scheduler (DDPM)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # Logger
        self.logger = Logger(self.save_dir / f"lora_rank{lora_rank}_train.log")
        
        # Mixed precision (fp16 또는 bf16 지원)
        mixed_precision_mode = config.get("mixed_precision", "fp16")
        self.use_amp = mixed_precision_mode in ("fp16", "bf16")
        self.scaler = None
        self.autocast_dtype = None

        if self.use_amp:
            if mixed_precision_mode == "bf16":
                # BF16은 GradScaler 없이도 안정적 (NaN 문제 해결 목적)
                self.autocast_dtype = torch.bfloat16
            else: # fp16 모드
                self.scaler = torch.cuda.amp.GradScaler()
                self.autocast_dtype = torch.float16
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """1 스텝 학습"""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # 랜덤 timestep 샘플링
        batch_size = images.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # 노이즈 추가
        noise = torch.randn_like(images)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        
        # 예측
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                noise_pred = self.model(
                    noisy_images,
                    timesteps,
                    class_labels=labels,
                    return_dict=False
                )[0]
                loss = F.mse_loss(noise_pred, noise)
        else:
            noise_pred = self.model(
                noisy_images,
                timesteps,
                class_labels=labels,
                return_dict=False
            )[0]
            loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def train_epoch(self, epoch: int) -> float:
        """1 에폭 학습"""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"LoRA Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            loss = self.train_step(batch)
            
            # Backward
            if self.use_amp:
                # Scaler 유무에 따라 backward 방식 분기
                if self.scaler is not None: # fp16 모드
                    self.scaler.scale(loss).backward()
                else: # bf16 모드 (scaler=None)
                    loss.backward()
                
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    # Scaler 유무에 따라 step 방식 분기
                    if self.scaler is not None: # fp16 모드
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else: # bf16 모드 (scaler=None)
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                loss.backward()
                
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            losses.update(loss.item())
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
            
            # 주기적 저장
            if (step + 1) % self.config["save_steps"] == 0:
                self.save_checkpoint(epoch, step)
        
        return losses.avg
    
    def save_checkpoint(self, epoch: int, step: int):
        """체크포인트 저장"""
        save_path = self.save_dir / f"lora_rank{self.lora_rank}_epoch{epoch}_step{step}.pt"
        self.model.save_pretrained(save_path)
        self.logger.log(f"Checkpoint saved to {save_path}")
    
    def train(self):
        """전체 학습 루프"""
        num_epochs = self.config["num_epochs"]
        
        self.logger.log(f"Starting LoRA training (rank={self.lora_rank})")
        self.logger.log(f"Train size: {len(self.train_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            
            self.logger.log(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            # 에폭 단위 저장
            if epoch % 10 == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, -1)
        
        self.logger.log("LoRA training completed")
        self.logger.close()


def train_lora(
    base_model: nn.Module,
    train_loader: DataLoader,
    lora_rank: int,
    save_dir: Optional[Path] = None,
    config: dict = LORA_CONFIG,
    seed: int = SEED
) -> nn.Module:
    """
    LoRA 파인튜닝 래퍼 함수
    
    Returns:
        LoRA가 적용된 모델
    """
    set_seed(seed)
    
    trainer = LoRATrainer(
        model=base_model,
        train_loader=train_loader,
        lora_rank=lora_rank,
        config=config,
        save_dir=save_dir
    )
    
    trainer.train()
    
    return trainer.model


def prepare_lora_training_data(
    images: np.ndarray,
    labels: np.ndarray,
    target_size: int = 64,  # ImageNet 64x64 모델에 맞춤
    batch_size: int = 16,
    num_workers: int = 4
) -> DataLoader:
    """
    LoRA 학습용 데이터로더 생성 (CIFAR-10 32x32 -> 64x64 업샘플링)
    
    Args:
        images: (N, H, W, C) numpy array, range [0, 1]
        labels: (N,) numpy array
        target_size: 타겟 이미지 크기 (64x64)
    """
    from torch.utils.data import TensorDataset
    import torch.nn.functional as F
    
    # Tensor로 변환
    images_tensor = torch.from_numpy(images).float()
    
    # (N, H, W, C) -> (N, C, H, W)
    if images_tensor.dim() == 4 and images_tensor.size(-1) == 3:
        images_tensor = images_tensor.permute(0, 3, 1, 2)
    
    # 32x32 -> 64x64 업샘플링
    if images_tensor.size(-1) != target_size:
        images_tensor = F.interpolate(
            images_tensor, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
    
    # [-1, 1] 범위로 정규화
    images_tensor = images_tensor * 2.0 - 1.0
    
    labels_tensor = torch.from_numpy(labels).long()
    
    dataset = TensorDataset(images_tensor, labels_tensor)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":
    print("LoRA training module loaded.")
    print("Note: Requires pretrained diffusion model for actual training.")