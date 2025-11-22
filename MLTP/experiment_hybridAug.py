"""
목적2: 전통적 증강 포화점 탐색 및 생성형 증강 추가효과 평가

실험 구성:
1. 전통적 증강을 k×100장씩 증가 (k=1~20)
2. 각 단계에서 ResNet-18 학습 및 성능 측정
3. 포화점 판단
4. 포화점 이후 생성형 증강 추가 (k×100장, k=1~10)
5. 성능 개선 확인
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from config import (
    SATURATION_EXP_CONFIG, CLASSIFIER_CONFIG, RESULTS_DIR,
    GENERATED_DIR, CHECKPOINT_DIR, DEVICE, SEED, CIFAR10_CONFIG
)
from data import (
    CIFAR10FewShot, get_base_transform, get_dataloader,
    AugmentedDataset, merge_datasets, extract_images_and_labels
)
from augment_traditional import generate_k_augmentations
from train_classifier import train_classifier
from gen_img2img import generate_augmented_images
from utils import (
    set_seed, save_results, plot_saturation_curve, Logger,
    plot_comparison_bar
)


class HybridAugExperiment:
    """목적2 실험 관리"""
    
    def __init__(
        self,
        best_generative_model,
        samples_per_class: int = 100,
        seed: int = SEED
    ):
        self.best_model = best_generative_model
        self.samples_per_class = samples_per_class
        self.seed = seed
        
        set_seed(seed)
        
        # 실험 디렉토리
        self.exp_dir = RESULTS_DIR / "objective2"
        self.exp_dir.mkdir(exist_ok=True, parents=True)
        
        # Logger
        self.logger = Logger(self.exp_dir / "experiment.log")
        
        # Few-shot 데이터 준비
        self.few_shot = CIFAR10FewShot(samples_per_class=samples_per_class, seed=seed)
        self.original_dataset = self.few_shot.get_few_shot_dataset(transform=None)
        self.original_images, self.original_labels = extract_images_and_labels(
            self.original_dataset
        )
        
        self.logger.log(f"Original dataset: {len(self.original_images)} images")
        
        # 결과 저장
        self.results = {
            'traditional_augmentation': {},
            'saturation_point': None,
            'generative_augmentation': {},
        }
    
    def find_saturation_point(self) -> int:
        """전통적 증강의 포화점 찾기"""
        self.logger.log("\n=== Phase 1: Finding Saturation Point ===")
        
        k_range = SATURATION_EXP_CONFIG["traditional_aug_range"]
        threshold = SATURATION_EXP_CONFIG["saturation_threshold"]
        window = SATURATION_EXP_CONFIG["saturation_window"]
        
        accuracies = []
        deltas = []
        
        for k in k_range:
            self.logger.log(f"\n--- k={k} (Traditional augmentation) ---")
            
            # k배 증강 생성
            aug_images, aug_labels = generate_k_augmentations(
                self.original_images,
                self.original_labels,
                k=k,
                seed=self.seed + k  # 매번 다른 증강
            )
            
            # 원본 + 증강 병합
            merged_images, merged_labels = merge_datasets(
                self.original_images,
                self.original_labels,
                aug_images,
                aug_labels
            )
            
            self.logger.log(f"Total training samples: {len(merged_images)}")
            
            # 학습
            train_dataset = AugmentedDataset(
                merged_images, merged_labels, transform=get_base_transform(train=True)
            )
            test_dataset = self.few_shot.get_test_dataset(
                transform=get_base_transform(train=False)
            )
            
            train_loader = get_dataloader(
                train_dataset, batch_size=CLASSIFIER_CONFIG["batch_size"], shuffle=True
            )
            test_loader = get_dataloader(
                test_dataset, batch_size=CLASSIFIER_CONFIG["batch_size"], shuffle=False
            )
            
            model, best_acc = train_classifier(
                train_loader=train_loader,
                test_loader=test_loader,
                save_dir=self.exp_dir / "traditional" / f"k{k}",
                seed=self.seed
            )
            
            accuracies.append(best_acc)
            self.results['traditional_augmentation'][k] = best_acc
            
            # 성능 증가폭 계산
            if len(accuracies) > 1:
                delta = accuracies[-1] - accuracies[-2]
                deltas.append(delta)
                self.logger.log(f"k={k} accuracy: {best_acc:.2f}% (Δ={delta:.2f}%)")
            else:
                self.logger.log(f"k={k} accuracy: {best_acc:.2f}%")
            
            # 포화점 판단
            if len(deltas) >= window:
                recent_deltas = deltas[-window:]
                
                if all(d < threshold for d in recent_deltas):
                    saturation_k = k - window + 1
                    self.logger.log(f"\n!!! Saturation detected at k={saturation_k} !!!")
                    self.logger.log(f"Recent deltas: {recent_deltas}")
                    
                    self.results['saturation_point'] = {
                        'k': saturation_k,
                        'accuracy': accuracies[saturation_k - 1],
                        'recent_deltas': recent_deltas
                    }
                    
                    return saturation_k
        
        # 포화점을 찾지 못한 경우 마지막 k 반환
        saturation_k = k_range[-1]
        self.logger.log(f"\nNo clear saturation found. Using k={saturation_k}")
        
        self.results['saturation_point'] = {
            'k': saturation_k,
            'accuracy': accuracies[-1],
            'note': 'No saturation detected'
        }
        
        return saturation_k
    
    def evaluate_generative_augmentation(self, saturation_k: int):
        """포화점 이후 생성형 증강 추가"""
        self.logger.log("\n=== Phase 2: Generative Augmentation Addition ===")
        
        # 포화점에서의 전통적 증강 데이터 생성
        self.logger.log(f"Generating traditional augmentation at saturation (k={saturation_k})...")
        trad_aug_images, trad_aug_labels = generate_k_augmentations(
            self.original_images,
            self.original_labels,
            k=saturation_k,
            seed=self.seed
        )
        
        # 포화점 데이터셋 (원본 + 전통적증강)
        saturation_images, saturation_labels = merge_datasets(
            self.original_images,
            self.original_labels,
            trad_aug_images,
            trad_aug_labels
        )
        
        self.logger.log(f"Saturation dataset size: {len(saturation_images)}")
        
        # 생성형 증강 데이터 1000장/class 생성
        self.logger.log("\nGenerating synthetic data with best model...")
        gen_images, gen_labels = generate_augmented_images(
            model=self.best_model,
            original_images=self.original_images,
            original_labels=self.original_labels,
            num_samples_per_class=1000,
            num_classes=CIFAR10_CONFIG["num_classes"],
            noise_level=50,
            guidance_scale=2.0,
            batch_size=16,
            seed=self.seed,
            save_path=GENERATED_DIR / "objective2_generative.pkl"
        )
        
        self.logger.log(f"Generated {len(gen_images)} synthetic images")
        
        # k×100장씩 추가하며 실험
        k_range = SATURATION_EXP_CONFIG["generative_aug_range"]
        
        for k in k_range:
            self.logger.log(f"\n--- k={k} (Generative augmentation) ---")
            
            # k×100장 샘플링
            samples_per_class = k * 100
            total_samples = samples_per_class * CIFAR10_CONFIG["num_classes"]
            
            # 클래스별로 균등하게 샘플링
            selected_indices = []
            for class_idx in range(CIFAR10_CONFIG["num_classes"]):
                class_mask = gen_labels == class_idx
                class_indices = np.where(class_mask)[0]
                
                selected = np.random.choice(
                    class_indices, size=samples_per_class, replace=False
                )
                selected_indices.extend(selected.tolist())
            
            selected_gen_images = gen_images[selected_indices]
            selected_gen_labels = gen_labels[selected_indices]
            
            # 포화점 데이터 + 생성형 증강 병합
            final_images, final_labels = merge_datasets(
                saturation_images,
                saturation_labels,
                selected_gen_images,
                selected_gen_labels
            )
            
            self.logger.log(f"Total training samples: {len(final_images)}")
            
            # 학습
            train_dataset = AugmentedDataset(
                final_images, final_labels, transform=get_base_transform(train=True)
            )
            test_dataset = self.few_shot.get_test_dataset(
                transform=get_base_transform(train=False)
            )
            
            train_loader = get_dataloader(
                train_dataset, batch_size=CLASSIFIER_CONFIG["batch_size"], shuffle=True
            )
            test_loader = get_dataloader(
                test_dataset, batch_size=CLASSIFIER_CONFIG["batch_size"], shuffle=False
            )
            
            model, best_acc = train_classifier(
                train_loader=train_loader,
                test_loader=test_loader,
                save_dir=self.exp_dir / "generative" / f"k{k}",
                seed=self.seed
            )
            
            self.results['generative_augmentation'][k] = best_acc
            self.logger.log(f"k={k} accuracy: {best_acc:.2f}%")
    
    def run_full_experiment(self):
        """전체 실험 실행"""
        self.logger.log("=" * 60)
        self.logger.log("Objective 2: Hybrid Augmentation Experiment")
        self.logger.log("=" * 60)
        
        # Phase 1: 포화점 찾기
        saturation_k = self.find_saturation_point()
        
        # Phase 2: 생성형 증강 추가
        self.evaluate_generative_augmentation(saturation_k)
        
        # 결과 저장
        save_results(self.results, self.exp_dir / "results.json")
        
        # 시각화
        self.plot_results()
        
        self.logger.log("\n=== Experiment completed ===")
        self.logger.close()
        
        return self.results
    
    def plot_results(self):
        """결과 시각화"""
        # 전통적 증강 포화 곡선
        trad_k = list(self.results['traditional_augmentation'].keys())
        trad_acc = list(self.results['traditional_augmentation'].values())
        
        saturation_k = self.results['saturation_point']['k'] if \
            self.results['saturation_point'] else None
        
        plot_saturation_curve(
            x_values=trad_k,
            y_values=trad_acc,
            save_path=self.exp_dir / "traditional_saturation.png",
            xlabel="Augmentation Multiplier (k)",
            ylabel="Accuracy (%)",
            title="Traditional Augmentation Saturation Curve",
            saturation_point=saturation_k
        )
        
        # 생성형 증강 효과
        if self.results['generative_augmentation']:
            gen_k = list(self.results['generative_augmentation'].keys())
            gen_acc = list(self.results['generative_augmentation'].values())
            
            plot_saturation_curve(
                x_values=gen_k,
                y_values=gen_acc,
                save_path=self.exp_dir / "generative_addition.png",
                xlabel="Generative Augmentation Multiplier (k)",
                ylabel="Accuracy (%)",
                title="Generative Augmentation Addition Effect"
            )
            
            # 비교 막대 그래프
            saturation_acc = self.results['saturation_point']['accuracy']
            max_gen_acc = max(gen_acc)
            
            methods = ["Saturation Point", "Best w/ Generative"]
            accuracies = [saturation_acc, max_gen_acc]
            
            plot_comparison_bar(
                methods, accuracies,
                save_path=self.exp_dir / "comparison.png",
                title="Saturation vs Generative Augmentation"
            )


if __name__ == "__main__":
    print("Objective 2 experiment module loaded.")
    print("Note: Requires best generative model from Objective 1.")