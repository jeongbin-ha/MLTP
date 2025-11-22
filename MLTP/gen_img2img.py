"""
Diffusion 모델 기반 img2img 생성
DDIM sampling을 사용한 이미지 변환
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm
from pathlib import Path
import pickle

from diffusers import DDIMScheduler
from config import DIFFUSION_CONFIG, DEVICE, SEED, GENERATED_DIR
from utils import set_seed


class Img2ImgGenerator:
    """Diffusion 기반 img2img 생성기"""
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: DDIMScheduler,
        noise_level: int = 50,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        device: str = DEVICE
    ):
        self.model = model.to(device)
        self.model.eval()
        self.noise_scheduler = noise_scheduler
        self.noise_level = noise_level
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        
        # DDIM timesteps 설정
        self.noise_scheduler.set_timesteps(num_inference_steps)
    
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Forward process: x0 -> xt
        이미지에 노이즈 추가
        
        Args:
            x0: 원본 이미지 (B, C, H, W)
            t: 노이즈 레벨 (timestep)
        
        Returns:
            xt: 노이즈가 추가된 이미지
        """
        noise = torch.randn_like(x0)
        timesteps = torch.full((x0.size(0),), t, device=x0.device, dtype=torch.long)
        
        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)
        return xt
    
    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        class_labels: torch.Tensor,
        start_step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reverse process: xt -> x0
        DDIM denoising
        
        Args:
            xt: 노이즈가 추가된 이미지
            class_labels: 클래스 레이블 (classifier-free guidance용)
            start_step: 시작 timestep (None이면 noise_level 사용)
        
        Returns:
            x0: 복원된 이미지
        """
        if start_step is None:
            start_step = self.noise_level
        
        # timesteps 재설정
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        
        # start_step에 가장 가까운 timestep 찾기
        start_idx = torch.argmin(torch.abs(timesteps - start_step)).item()
        
        x = xt
        
        # Denoising loop
        for i in range(start_idx, len(timesteps)):
            t = timesteps[i]
            t_batch = t.repeat(x.size(0)).to(self.device)
            
            # Classifier-free guidance
            if self.guidance_scale > 1.0:
                # Conditional prediction
                noise_pred_cond = self.model(
                    x, t_batch, class_labels=class_labels, return_dict=False
                )[0]
                
                # Unconditional prediction (null class)
                null_labels = torch.full_like(class_labels, -1)
                noise_pred_uncond = self.model(
                    x, t_batch, class_labels=null_labels, return_dict=False
                )[0]
                
                # Guidance
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = self.model(
                    x, t_batch, class_labels=class_labels, return_dict=False
                )[0]
            
            # DDIM step
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        
        return x
    
    @torch.no_grad()
    def img2img(
        self,
        images: torch.Tensor,
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        img2img 변환: q_sample -> p_sample
        
        Args:
            images: 입력 이미지 (B, C, H, W), range [-1, 1]
            class_labels: 클래스 레이블
        
        Returns:
            변환된 이미지 (B, C, H, W), range [-1, 1]
        """
        images = images.to(self.device)
        class_labels = class_labels.to(self.device)
        
        # Forward: x0 -> xt
        xt = self.q_sample(images, self.noise_level)
        
        # Reverse: xt -> x0
        x0 = self.p_sample(xt, class_labels)
        
        return x0
    
    @torch.no_grad()
    def generate_batch(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 16,
        model_size: int = 64,  # 모델 입력 크기
        output_size: int = 32  # 최종 출력 크기 (CIFAR-10)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 단위로 이미지 생성
        32x32 입력 -> 64x64 업샘플 -> 생성 -> 32x32 다운샘플
        
        Args:
            images: (N, H, W, C) numpy array, range [0, 1]
            labels: (N,) numpy array
            batch_size: 배치 크기
            model_size: 모델 학습 해상도 (64)
            output_size: 최종 출력 해상도 (32)
        
        Returns:
            generated_images: (N, H, W, C), range [0, 1]
            generated_labels: (N,)
        """
        import torch.nn.functional as F
        
        # Tensor로 변환 및 정규화
        images_tensor = torch.from_numpy(images).float()
        
        # (N, H, W, C) -> (N, C, H, W)
        if images_tensor.dim() == 4 and images_tensor.size(-1) == 3:
            images_tensor = images_tensor.permute(0, 3, 1, 2)
        
        # 32x32 -> 64x64 업샘플링
        if images_tensor.size(-1) != model_size:
            images_tensor = F.interpolate(
                images_tensor, 
                size=(model_size, model_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # [0, 1] -> [-1, 1]
        images_tensor = images_tensor * 2.0 - 1.0
        
        labels_tensor = torch.from_numpy(labels).long()
        
        # 배치 단위 생성
        generated = []
        num_samples = len(images_tensor)
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating"):
            batch_imgs = images_tensor[i:i+batch_size]
            batch_labels = labels_tensor[i:i+batch_size]
            
            gen_imgs = self.img2img(batch_imgs, batch_labels)
            
            # 64x64 -> 32x32 다운샘플링
            if output_size != model_size:
                gen_imgs = F.interpolate(
                    gen_imgs, 
                    size=(output_size, output_size),
                    mode='bilinear', 
                    align_corners=False
                )
            
            generated.append(gen_imgs.cpu())
        
        generated = torch.cat(generated, dim=0)
        
        # [-1, 1] -> [0, 1]
        generated = (generated + 1.0) / 2.0
        generated = torch.clamp(generated, 0.0, 1.0)
        
        # (N, C, H, W) -> (N, H, W, C)
        generated = generated.permute(0, 2, 3, 1).numpy()
        
        return generated, labels


def generate_augmented_images(
    model: nn.Module,
    original_images: np.ndarray,
    original_labels: np.ndarray,
    num_samples_per_class: int,
    num_classes: int = 10,
    noise_level: int = 50,
    guidance_scale: float = 2.0,
    batch_size: int = 16,
    model_size: int = 64,  # 모델 학습 해상도
    output_size: int = 32,  # CIFAR-10 출력 해상도
    seed: int = SEED,
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    클래스 균형을 유지하며 생성형 증강 이미지 생성
    
    Args:
        model: Diffusion 모델
        original_images: 원본 이미지
        original_labels: 원본 레이블
        num_samples_per_class: 클래스당 생성할 샘플 수
        num_classes: 클래스 개수
        noise_level: 노이즈 레벨
        guidance_scale: Classifier-free guidance scale
        batch_size: 배치 크기
        model_size: 모델 입력 해상도 (64)
        output_size: 최종 출력 해상도 (32)
        seed: 랜덤 시드
        save_path: 저장 경로
    
    Returns:
        generated_images, generated_labels
    """
    set_seed(seed)
    
    # Scheduler 초기화
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    
    # Generator 생성
    generator = Img2ImgGenerator(
        model=model,
        noise_scheduler=noise_scheduler,
        noise_level=noise_level,
        num_inference_steps=50,
        guidance_scale=guidance_scale
    )
    
    all_generated = []
    all_labels = []
    
    # 클래스별로 생성
    for class_idx in range(num_classes):
        print(f"\nGenerating class {class_idx}...")
        
        # 해당 클래스의 원본 이미지 선택
        class_mask = original_labels == class_idx
        class_images = original_images[class_mask]
        
        if len(class_images) == 0:
            print(f"Warning: No images for class {class_idx}")
            continue
        
        # 필요한 만큼 반복 샘플링
        generated_count = 0
        while generated_count < num_samples_per_class:
            # 배치 크기만큼 샘플링
            sample_size = min(batch_size, num_samples_per_class - generated_count)
            
            # 랜덤하게 원본 이미지 선택
            indices = np.random.choice(len(class_images), size=sample_size, replace=True)
            batch_images = class_images[indices]
            batch_labels = np.full(sample_size, class_idx)
            
            # 생성 (업샘플링 -> 생성 -> 다운샘플링)
            gen_images, gen_labels = generator.generate_batch(
                batch_images, batch_labels, 
                batch_size=sample_size,
                model_size=model_size,
                output_size=output_size
            )
            
            all_generated.append(gen_images)
            all_labels.append(gen_labels)
            
            generated_count += sample_size
    
    # 결합
    generated_images = np.concatenate(all_generated, axis=0)
    generated_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal generated: {len(generated_images)} images")
    print(f"Output shape: {generated_images.shape}")
    
    # 저장
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'images': generated_images,
                'labels': generated_labels,
                'noise_level': noise_level,
                'guidance_scale': guidance_scale,
                'num_samples_per_class': num_samples_per_class,
                'model_size': model_size,
                'output_size': output_size
            }, f)
        print(f"Saved to {save_path}")
    
    return generated_images, generated_labels


if __name__ == "__main__":
    print("img2img generation module loaded.")
    print("Note: Requires trained diffusion model for generation.")