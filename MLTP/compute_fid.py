"""
FID (Frechet Inception Distance) 계산
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3
from scipy import linalg
from typing import Tuple
from tqdm import tqdm

from config import FID_CONFIG, DEVICE


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 feature 추출기"""
    
    def __init__(self, device: str = DEVICE):
        super().__init__()
        
        # InceptionV3 로드 (pretrained)
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # FC layer 전까지만 사용
        self.inception = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        self.inception = self.inception.to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor, range [0, 1]
        
        Returns:
            features: (B, 2048)
        """
        # 32x32 -> 299x299 (InceptionV3 입력 크기)
        x = torch.nn.functional.interpolate(
            x, size=(299, 299), mode='bilinear', align_corners=False
        )
        
        # [0, 1] -> [-1, 1]
        x = x * 2.0 - 1.0
        
        features = self.inception(x)
        features = features.squeeze(-1).squeeze(-1)
        
        return features


def extract_features(
    images: np.ndarray,
    batch_size: int = 50,
    device: str = DEVICE
) -> np.ndarray:
    """
    이미지에서 InceptionV3 features 추출
    
    Args:
        images: (N, H, W, C) numpy array, range [0, 1]
        batch_size: 배치 크기
        device: 디바이스
    
    Returns:
        features: (N, 2048) numpy array
    """
    # Feature extractor 생성
    extractor = InceptionV3FeatureExtractor(device=device)
    extractor.eval()
    
    # Tensor로 변환
    images_tensor = torch.from_numpy(images).float()
    
    # (N, H, W, C) -> (N, C, H, W)
    if images_tensor.dim() == 4 and images_tensor.size(-1) == 3:
        images_tensor = images_tensor.permute(0, 3, 1, 2)
    
    # 데이터로더 생성
    dataset = TensorDataset(images_tensor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Feature 추출
    all_features = []
    
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            features = extractor(batch)
            all_features.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    
    return all_features


def calculate_fid_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    평균과 공분산 계산
    
    Args:
        features: (N, D) numpy array
    
    Returns:
        mu: (D,) 평균
        sigma: (D, D) 공분산
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Frechet Distance 계산
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Means have different shapes"
    assert sigma1.shape == sigma2.shape, "Covariances have different shapes"
    
    diff = mu1 - mu2
    
    # 공분산 행렬의 곱 계산
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 수치 안정성
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 허수부 제거
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


def compute_fid(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    batch_size: int = 50,
    device: str = DEVICE
) -> float:
    """
    FID 계산
    
    Args:
        real_images: (N, H, W, C) numpy array, range [0, 1]
        fake_images: (M, H, W, C) numpy array, range [0, 1]
        batch_size: 배치 크기
        device: 디바이스
    
    Returns:
        fid: FID 값
    """
    print("Computing FID...")
    
    # Real features 추출
    print("Extracting real features...")
    real_features = extract_features(real_images, batch_size, device)
    
    # Fake features 추출
    print("Extracting fake features...")
    fake_features = extract_features(fake_images, batch_size, device)
    
    # 통계량 계산
    mu_real, sigma_real = calculate_fid_statistics(real_features)
    mu_fake, sigma_fake = calculate_fid_statistics(fake_features)
    
    # FID 계산
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    print(f"FID: {fid:.2f}")
    
    return fid


def compute_fid_from_datasets(
    real_dataset,
    fake_dataset,
    batch_size: int = 50,
    device: str = DEVICE
) -> float:
    """
    Dataset에서 직접 FID 계산
    """
    # Real 이미지 추출
    real_images = []
    for img, _ in tqdm(real_dataset, desc="Loading real images"):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        else:
            img = np.array(img)
        real_images.append(img)
    real_images = np.array(real_images)
    
    # Fake 이미지 추출
    fake_images = []
    for img, _ in tqdm(fake_dataset, desc="Loading fake images"):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        else:
            img = np.array(img)
        fake_images.append(img)
    fake_images = np.array(fake_images)
    
    # FID 계산
    return compute_fid(real_images, fake_images, batch_size, device)


if __name__ == "__main__":
    # 테스트
    print("Testing FID computation...")
    
    # 더미 데이터
    real = np.random.rand(100, 32, 32, 3).astype(np.float32)
    fake = np.random.rand(100, 32, 32, 3).astype(np.float32)
    
    fid = compute_fid(real, fake, batch_size=10)
    print(f"Test FID: {fid:.2f}")