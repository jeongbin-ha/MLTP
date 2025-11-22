"""
공통 유틸리티 함수
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    accuracy: float,
    save_path: Path,
    is_best: bool = False
):
    """모델 체크포인트 저장"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
        "accuracy": accuracy,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / f"{save_path.stem}_best.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "accuracy": checkpoint["accuracy"],
    }


def save_results(results: Dict[str, Any], save_path: Path):
    """실험 결과를 JSON으로 저장"""
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def load_results(load_path: Path) -> Dict[str, Any]:
    """JSON 결과 로드"""
    with open(load_path, "r") as f:
        return json.load(f)


class AverageMeter:
    """평균 및 현재값 추적"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Top-k 정확도 계산"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res[0] if len(res) == 1 else res


def plot_training_curves(
    train_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Path
):
    """학습 곡선 시각화"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(
    methods: List[str],
    accuracies: List[float],
    save_path: Path,
    title: str = "Model Comparison",
    ylabel: str = "Accuracy (%)"
):
    """모델 비교 막대 그래프"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, color='steelblue', alpha=0.8)
    
    # 각 막대 위에 값 표시
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_saturation_curve(
    x_values: List[int],
    y_values: List[float],
    save_path: Path,
    xlabel: str = "Augmentation Multiplier (k)",
    ylabel: str = "Accuracy (%)",
    title: str = "Augmentation Saturation Curve",
    saturation_point: Optional[int] = None
):
    """포화점 곡선 시각화"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_values, y_values, 'b-o', linewidth=2, markersize=6)
    
    if saturation_point:
        ax.axvline(x=saturation_point, color='r', linestyle='--', 
                   label=f'Saturation Point (k={saturation_point})')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if saturation_point:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_experiment_dir(base_dir: Path, exp_name: str) -> Path:
    """실험별 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def print_experiment_config(config: Dict[str, Any]):
    """실험 설정 출력"""
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("=" * 60)


class Logger:
    """간단한 로거"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'a')
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + '\n')
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()