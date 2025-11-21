"""
ResNet-18 분류기 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from tqdm import tqdm

from resnet import get_resnet18_cifar10
from config import CLASSIFIER_CONFIG, DEVICE, SEED, CHECKPOINT_DIR
from utils import (
    set_seed, save_checkpoint, AverageMeter, accuracy,
    plot_training_curves, Logger
)


class ClassifierTrainer:
    """ResNet-18 학습 관리"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: dict = CLASSIFIER_CONFIG,
        save_dir: Optional[Path] = None,
        device: str = DEVICE
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        if save_dir is None:
            save_dir = CHECKPOINT_DIR
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )
        
        # Scheduler
        if config["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config["num_epochs"]
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # 학습 기록
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        self.best_acc = 0.0
        
        # Logger
        self.logger = Logger(self.save_dir / "train.log")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """1 에폭 학습"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 측정
            acc = accuracy(outputs, targets)
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
        
        return losses.avg, top1.avg
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """테스트셋 평가"""
        self.model.eval()
        
        top1 = AverageMeter()
        
        for images, targets in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images)
            acc = accuracy(outputs, targets)
            top1.update(acc, images.size(0))
        
        return top1.avg
    
    def train(self, num_epochs: Optional[int] = None):
        """전체 학습 루프"""
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]
        
        self.logger.log(f"Starting training for {num_epochs} epochs")
        self.logger.log(f"Train size: {len(self.train_loader.dataset)}")
        self.logger.log(f"Test size: {len(self.test_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            # 학습
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 평가
            test_acc = self.evaluate()
            
            # Scheduler step
            self.scheduler.step()
            
            # 기록
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            
            # 로그
            self.logger.log(
                f"Epoch {epoch}/{num_epochs} - "
                f"Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Test Acc: {test_acc:.2f}%"
            )
            
            # 체크포인트 저장
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
            
            if epoch % 10 == 0 or is_best:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    test_acc,
                    self.save_dir / f"checkpoint_epoch_{epoch}.pth",
                    is_best=is_best
                )
        
        self.logger.log(f"Training completed. Best accuracy: {self.best_acc:.2f}%")
        
        # 학습 곡선 저장
        plot_training_curves(
            self.train_losses,
            self.train_accs,
            self.test_accs,
            self.save_dir / "training_curves.png"
        )
        
        self.logger.close()
        
        return self.best_acc


def train_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 10,
    save_dir: Optional[Path] = None,
    config: dict = CLASSIFIER_CONFIG,
    seed: int = SEED
) -> Tuple[nn.Module, float]:
    """
    분류기 학습 래퍼 함수
    
    Returns:
        trained_model, best_accuracy
    """
    set_seed(seed)
    
    # 모델 생성
    model = get_resnet18_cifar10(num_classes=num_classes)
    
    # 트레이너 생성 및 학습
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        save_dir=save_dir
    )
    
    best_acc = trainer.train()
    
    return model, best_acc


if __name__ == "__main__":
    from data import CIFAR10FewShot, get_base_transform, get_dataloader
    
    print("Testing classifier training...")
    
    # Few-shot 데이터 준비
    few_shot = CIFAR10FewShot(samples_per_class=100)
    
    train_dataset = few_shot.get_few_shot_dataset(
        transform=get_base_transform(train=True)
    )
    test_dataset = few_shot.get_test_dataset(
        transform=get_base_transform(train=False)
    )
    
    train_loader = get_dataloader(train_dataset, batch_size=128, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False)
    
    # 학습 (테스트용 5 에폭만)
    test_config = CLASSIFIER_CONFIG.copy()
    test_config["num_epochs"] = 5
    
    model, best_acc = train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        config=test_config,
        save_dir=CHECKPOINT_DIR / "test"
    )
    
    print(f"\nTest training completed. Best accuracy: {best_acc:.2f}%")