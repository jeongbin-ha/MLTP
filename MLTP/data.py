"""
CIFAR-10 데이터 로딩 및 few-shot 데이터셋 생성
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import pickle

from config import CIFAR10_CONFIG, DATA_DIR, SEED
from utils import set_seed


class CIFAR10FewShot:
    """CIFAR-10 few-shot 데이터셋 관리"""
    
    def __init__(
        self,
        root: Path = DATA_DIR,
        samples_per_class: int = 100,
        seed: int = SEED,
        download: bool = True
    ):
        self.root = root
        self.samples_per_class = samples_per_class
        self.seed = seed
        
        set_seed(seed)
        
        # CIFAR-10 다운로드 및 로드
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=None
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=None
        )
        
        # few-shot 인덱스 추출
        self.few_shot_indices = self._extract_few_shot_indices()
        
        # 저장 경로
        self.indices_path = root / f"few_shot_indices_{samples_per_class}shot.pkl"
        self._save_indices()
    
    def _extract_few_shot_indices(self) -> List[int]:
        """클래스당 N개씩 샘플 인덱스 추출"""
        targets = np.array(self.train_dataset.targets)
        indices = []
        
        for class_idx in range(CIFAR10_CONFIG["num_classes"]):
            class_indices = np.where(targets == class_idx)[0]
            
            # 랜덤 샘플링
            selected = np.random.choice(
                class_indices,
                size=self.samples_per_class,
                replace=False
            )
            indices.extend(selected.tolist())
        
        return sorted(indices)
    
    def _save_indices(self):
        """인덱스 저장"""
        with open(self.indices_path, 'wb') as f:
            pickle.dump(self.few_shot_indices, f)
        print(f"Few-shot indices saved to {self.indices_path}")
    
    def load_indices(self, indices_path: Optional[Path] = None) -> List[int]:
        """저장된 인덱스 로드"""
        if indices_path is None:
            indices_path = self.indices_path
        
        with open(indices_path, 'rb') as f:
            indices = pickle.load(f)
        
        print(f"Loaded {len(indices)} few-shot indices from {indices_path}")
        return indices
    
    def get_few_shot_dataset(self, transform=None) -> Subset:
        """Few-shot 서브셋 반환"""
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=False, transform=transform
        )
        return Subset(dataset, self.few_shot_indices)
    
    def get_full_train_dataset(self, transform=None):
        """전체 학습 데이터셋 반환 (FID 계산용)"""
        return torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=False, transform=transform
        )
    
    def get_test_dataset(self, transform=None):
        """테스트 데이터셋 반환"""
        return torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=False, transform=transform
        )
    
    def get_class_distribution(self) -> dict:
        """Few-shot 데이터셋의 클래스 분포 확인"""
        targets = np.array(self.train_dataset.targets)
        selected_targets = targets[self.few_shot_indices]
        
        distribution = {}
        for class_idx in range(CIFAR10_CONFIG["num_classes"]):
            count = np.sum(selected_targets == class_idx)
            distribution[class_idx] = count
        
        return distribution


def get_base_transform(train: bool = True):
    """기본 전처리 변환"""
    if train:
        # 학습시에는 최소한의 normalize만
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])
    else:
        # 테스트시
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """데이터로더 생성"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


class AugmentedDataset(Dataset):
    """증강 데이터를 포함하는 커스텀 데이터셋"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            images: (N, H, W, C) numpy array
            labels: (N,) numpy array
            transform: 적용할 변환
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # PIL Image로 변환 (transform 적용 위해)
        from PIL import Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def merge_datasets(
    original_images: np.ndarray,
    original_labels: np.ndarray,
    augmented_images: np.ndarray,
    augmented_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """원본과 증강 데이터를 병합"""
    merged_images = np.concatenate([original_images, augmented_images], axis=0)
    merged_labels = np.concatenate([original_labels, augmented_labels], axis=0)
    return merged_images, merged_labels


def extract_images_and_labels(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """데이터셋에서 이미지와 레이블 추출"""
    images = []
    labels = []
    
    for img, label in dataset:
        # PIL Image를 numpy array로 변환
        if hasattr(img, 'numpy'):
            img = img.numpy()
        else:
            img = np.array(img)
        
        images.append(img)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


if __name__ == "__main__":
    # 테스트
    print("Creating CIFAR-10 few-shot dataset...")
    few_shot = CIFAR10FewShot(samples_per_class=100)
    
    print("\nClass distribution:")
    distribution = few_shot.get_class_distribution()
    for class_idx, count in distribution.items():
        print(f"  Class {class_idx}: {count} samples")
    
    print(f"\nTotal few-shot samples: {len(few_shot.few_shot_indices)}")
    print(f"Test set size: {len(few_shot.test_dataset)}")