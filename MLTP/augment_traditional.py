"""
전통적 데이터 증강 (Random Crop, Flip, Color Jitter 등)
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from tqdm import tqdm
import pickle

from config import TRADITIONAL_AUG_CONFIG, DATA_DIR
from data import CIFAR10FewShot, extract_images_and_labels


def get_augmentation_transform(config: dict = TRADITIONAL_AUG_CONFIG):
    """전통적 증강 변환 생성"""
    transform_list = []
    
    # Random Crop
    if config.get("random_crop", True):
        transform_list.append(
            transforms.RandomCrop(
                config.get("crop_size", 32),
                padding=config.get("padding", 4)
            )
        )
    
    # Horizontal Flip
    if config.get("horizontal_flip", True):
        transform_list.append(
            transforms.RandomHorizontalFlip(p=config.get("flip_prob", 0.5))
        )
    
    # Color Jitter
    if config.get("color_jitter", True):
        transform_list.append(
            transforms.ColorJitter(
                brightness=config.get("brightness", 0.2),
                contrast=config.get("contrast", 0.2),
                saturation=config.get("saturation", 0.2),
                hue=config.get("hue", 0.1)
            )
        )
    
    return transforms.Compose(transform_list)


def augment_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    num_augmentations_per_sample: int,
    config: dict = TRADITIONAL_AUG_CONFIG,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    데이터셋에 전통적 증강 적용
    
    Args:
        images: (N, H, W, C) numpy array
        labels: (N,) numpy array
        num_augmentations_per_sample: 샘플당 생성할 증강 이미지 수
        config: 증강 설정
        seed: 랜덤 시드
    
    Returns:
        augmented_images: (N*num_aug, H, W, C)
        augmented_labels: (N*num_aug,)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    transform = get_augmentation_transform(config)
    
    augmented_images = []
    augmented_labels = []
    
    print(f"Generating {num_augmentations_per_sample} augmentations per sample...")
    
    for img, label in tqdm(zip(images, labels), total=len(images)):
        for _ in range(num_augmentations_per_sample):
            # numpy array를 PIL Image로 변환
            if img.dtype != np.uint8:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            pil_img = Image.fromarray(img_uint8)
            
            # 증강 적용
            aug_img = transform(pil_img)
            
            # numpy array로 다시 변환
            aug_img_np = np.array(aug_img)
            
            augmented_images.append(aug_img_np)
            augmented_labels.append(label)
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    return augmented_images, augmented_labels


def generate_traditional_augmentation(
    samples_per_class: int = 100,
    augmentations_per_sample: int = 5,
    save_path: Optional[str] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Few-shot 데이터셋에 전통적 증강 적용 및 저장
    
    Args:
        samples_per_class: 클래스당 원본 샘플 수
        augmentations_per_sample: 샘플당 증강 수
        save_path: 저장 경로
        seed: 랜덤 시드
    
    Returns:
        augmented_images, augmented_labels
    """
    # Few-shot 데이터 로드
    few_shot = CIFAR10FewShot(samples_per_class=samples_per_class, seed=seed)
    dataset = few_shot.get_few_shot_dataset(transform=None)
    
    # 이미지와 레이블 추출
    images, labels = extract_images_and_labels(dataset)
    
    print(f"Original dataset: {len(images)} images")
    print(f"Generating {augmentations_per_sample} augmentations per sample...")
    
    # 증강 적용
    aug_images, aug_labels = augment_dataset(
        images, labels, augmentations_per_sample, seed=seed
    )
    
    print(f"Generated {len(aug_images)} augmented images")
    
    # 저장
    if save_path is None:
        save_path = DATA_DIR / f"traditional_aug_{augmentations_per_sample}x.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'images': aug_images,
            'labels': aug_labels,
            'original_images': images,
            'original_labels': labels,
            'augmentations_per_sample': augmentations_per_sample,
            'samples_per_class': samples_per_class
        }, f)
    
    print(f"Saved to {save_path}")
    
    return aug_images, aug_labels


def load_traditional_augmentation(load_path: str) -> dict:
    """저장된 증강 데이터 로드"""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded traditional augmentation from {load_path}")
    print(f"  Original: {len(data['original_images'])} images")
    print(f"  Augmented: {len(data['images'])} images")
    
    return data


def generate_k_augmentations(
    original_images: np.ndarray,
    original_labels: np.ndarray,
    k: int,
    config: dict = TRADITIONAL_AUG_CONFIG,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    원본 데이터에 k배의 전통적 증강 생성 (목적2용)
    
    Args:
        original_images: 원본 이미지
        original_labels: 원본 레이블
        k: 증강 배수 (k=1이면 원본과 같은 수)
        config: 증강 설정
        seed: 랜덤 시드
    
    Returns:
        k배 증강된 이미지와 레이블
    """
    if k == 0:
        return np.array([]), np.array([])
    
    return augment_dataset(
        original_images,
        original_labels,
        num_augmentations_per_sample=k,
        config=config,
        seed=seed
    )


if __name__ == "__main__":
    # 테스트: 샘플당 5개씩 증강 생성
    print("Testing traditional augmentation...")
    aug_images, aug_labels = generate_traditional_augmentation(
        samples_per_class=100,
        augmentations_per_sample=5,
        seed=42
    )
    
    print(f"\nFinal shape:")
    print(f"  Images: {aug_images.shape}")
    print(f"  Labels: {aug_labels.shape}")
    
    # 클래스별 분포 확인
    unique, counts = np.unique(aug_labels, return_counts=True)
    print(f"\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")