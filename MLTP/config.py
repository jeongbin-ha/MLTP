"""
프로젝트 설정 및 하이퍼파라미터 중앙 관리
"""

import os
from pathlib import Path


# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
GENERATED_DIR = BASE_DIR / "generated"

# 디렉토리 생성
for dir_path in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, GENERATED_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


# CIFAR-10 설정
CIFAR10_CONFIG = {
    "num_classes": 10,
    "image_size": 32,
    "num_channels": 3,
    "samples_per_class": 100,  # few-shot 설정
    "test_size": 10000,
}


# ResNet-18 학습 설정
CLASSIFIER_CONFIG = {
    "batch_size": 128,
    "num_epochs": 100,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "cosine",  # cosine annealing
    "warmup_epochs": 5,
    "num_workers": 4,
}


# 전통적 증강 설정
TRADITIONAL_AUG_CONFIG = {
    "random_crop": True,
    "crop_size": 32,
    "padding": 4,
    "horizontal_flip": True,
    "flip_prob": 0.5,
    "color_jitter": True,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "normalize": True,
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2470, 0.2435, 0.2616],
}


# Diffusion 모델 설정
DIFFUSION_CONFIG = {
    "model_name": "openai/diffusers-ct_imagenet64",  # ImageNet 64x64 모델
    "model_size": 64,  # 모델 학습 해상도
    "target_size": 32,  # CIFAR-10 타겟 해상도
    "ddim_steps": 50,
    "noise_level": 50,  # t value for img2img
    "guidance_scale_min": 1.5,
    "guidance_scale_max": 3.0,
}


# LoRA 파인튜닝 설정
LORA_CONFIG = {
    "ranks": [4, 8, 16, 32],
    "alpha_scale": 1.0,  # LoRA scaling factor
    "dropout": 0.1,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],  # UNet attention layers
    
    # 파인튜닝 하이퍼파라미터
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 5e-5,
    "weight_decay": 0.0,
    "warmup_steps": 500,
    "gradient_accumulation_steps": 8,
    "mixed_precision": "bf16",
    "save_steps": 1000,
}


# 생성 실험 설정 (목적1)
GENERATION_EXP_CONFIG = {
    "samples_per_class": 500,  # 클래스당 생성할 이미지 수
    "total_samples": 5000,  # 총 생성 이미지 수
    "seed": 42,
}


# FID 계산 설정
FID_CONFIG = {
    "batch_size": 50,
    "dims": 2048,  # InceptionV3 feature dimension
    "num_workers": 4,
}


# 포화점 실험 설정 (목적2)
SATURATION_EXP_CONFIG = {
    "traditional_aug_range": list(range(1, 21)),  # k=1~20
    "generative_aug_range": list(range(1, 11)),  # k=1~10
    "saturation_threshold": 0.005,  # 성능 증가폭이 0.5% 미만이면 포화로 판단
    "saturation_window": 3,  # 연속 3번 threshold 이하면 포화
}


# 실험 재현성
SEED = 42


# 디바이스 설정
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0