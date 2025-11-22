"""
목적1: 생성모델 기반 증강 성능 검증 및 최적 조건 탐색

실험 구성:
1. 모델군 9개 준비 (Non-finetuned 1개 + LoRA 8개)
2. Baseline 성능 측정 (원본만, 원본+전통적증강)
3. 각 생성모델로 증강 데이터 생성 및 분류 성능 측정
4. FID 계산
5. 최적 모델 선정
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

from config import (
    LORA_CONFIG, GENERATION_EXP_CONFIG, CLASSIFIER_CONFIG, RESULTS_DIR,
    GENERATED_DIR, CHECKPOINT_DIR, DEVICE, SEED, CIFAR10_CONFIG
)
from data import (
    CIFAR10FewShot, get_base_transform, get_dataloader,
    AugmentedDataset, merge_datasets, extract_images_and_labels
)
from augment_traditional import generate_k_augmentations
from train_classifier import train_classifier
from train_lora import train_lora, prepare_lora_training_data
from gen_img2img import generate_augmented_images
from compute_fid import compute_fid
from utils import set_seed, save_results, plot_comparison_bar, Logger


class OptimalGenExperiment:
    """목적1 실험 관리"""
    
    def __init__(
        self,
        base_diffusion_model,
        samples_per_class: int = 100,
        seed: int = SEED
    ):
        self.base_diffusion_model = base_diffusion_model
        self.samples_per_class = samples_per_class
        self.seed = seed
        
        set_seed(seed)
        
        # 실험 디렉토리
        self.exp_dir = RESULTS_DIR / "objective1"
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
            'baselines': {},
            'generative_models': {},
            'fid_scores': {},
            'best_model': None
        }
    
    def prepare_traditional_augmentation(self, augmentations_per_sample: int = 5):
        """전통적 증강 데이터 준비"""
        self.logger.log(f"Preparing traditional augmentation ({augmentations_per_sample}x)...")
        
        self.trad_aug_images, self.trad_aug_labels = generate_k_augmentations(
            self.original_images,
            self.original_labels,
            k=augmentations_per_sample,
            seed=self.seed
        )
        
        self.logger.log(f"Generated {len(self.trad_aug_images)} traditional augmented images")
    
    def run_baseline1(self) -> float:
        """Baseline 1: 원본 100장만 학습"""
        self.logger.log("\n=== Baseline 1: Original data only ===")
        
        train_dataset = self.few_shot.get_few_shot_dataset(
            transform=get_base_transform(train=True)
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
            save_dir=self.exp_dir / "baseline1",
            seed=self.seed
        )
        
        self.results['baselines']['original_only'] = best_acc
        self.logger.log(f"Baseline 1 accuracy: {best_acc:.2f}%")
        
        return best_acc
    
    def run_baseline2(self) -> float:
        """Baseline 2: 원본 + 전통적 증강"""
        self.logger.log("\n=== Baseline 2: Original + Traditional augmentation ===")
        
        # 데이터 병합
        merged_images, merged_labels = merge_datasets(
            self.original_images,
            self.original_labels,
            self.trad_aug_images,
            self.trad_aug_labels
        )
        
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
            save_dir=self.exp_dir / "baseline2",
            seed=self.seed
        )
        
        self.results['baselines']['original_plus_traditional'] = best_acc
        self.logger.log(f"Baseline 2 accuracy: {best_acc:.2f}%")
        
        return best_acc
    
    def train_lora_models(self):
        """LoRA 모델들 학습"""
        self.logger.log("\n=== Training LoRA models ===")
        
        lora_ranks = LORA_CONFIG["ranks"]
        
        # Case A: 원본만
        self.logger.log("\nCase A: Training with original data only")
        case_a_loader = prepare_lora_training_data(
            self.original_images,
            self.original_labels,
            batch_size=LORA_CONFIG["batch_size"]
        )
        
        for rank in lora_ranks:
            self.logger.log(f"\n  Training LoRA rank={rank} (Case A)")
            model = train_lora(
                base_model=self.base_diffusion_model,
                train_loader=case_a_loader,
                lora_rank=rank,
                save_dir=self.exp_dir / "lora" / f"case_a_rank{rank}",
                seed=self.seed
            )
            
            # 모델 저장
            save_path = CHECKPOINT_DIR / f"lora_case_a_rank{rank}.pt"
            torch.save(model.state_dict(), save_path)
        
        # Case B: 원본 + 전통적 증강
        self.logger.log("\nCase B: Training with original + traditional augmentation")
        merged_images, merged_labels = merge_datasets(
            self.original_images,
            self.original_labels,
            self.trad_aug_images,
            self.trad_aug_labels
        )
        
        case_b_loader = prepare_lora_training_data(
            merged_images,
            merged_labels,
            batch_size=LORA_CONFIG["batch_size"]
        )
        
        for rank in lora_ranks:
            self.logger.log(f"\n  Training LoRA rank={rank} (Case B)")
            model = train_lora(
                base_model=self.base_diffusion_model,
                train_loader=case_b_loader,
                lora_rank=rank,
                save_dir=self.exp_dir / "lora" / f"case_b_rank{rank}",
                seed=self.seed
            )
            
            # 모델 저장
            save_path = CHECKPOINT_DIR / f"lora_case_b_rank{rank}.pt"
            torch.save(model.state_dict(), save_path)
    
    def generate_with_model(
        self,
        model,
        model_name: str,
        samples_per_class: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """특정 모델로 생성 데이터 생성"""
        self.logger.log(f"\nGenerating with {model_name}...")
        
        gen_images, gen_labels = generate_augmented_images(
            model=model,
            original_images=self.original_images,
            original_labels=self.original_labels,
            num_samples_per_class=samples_per_class,
            num_classes=CIFAR10_CONFIG["num_classes"],
            noise_level=50,
            guidance_scale=2.0,
            batch_size=16,
            seed=self.seed,
            save_path=GENERATED_DIR / f"{model_name}.pkl"
        )
        
        return gen_images, gen_labels
    
    def evaluate_generative_model(
        self,
        model,
        model_name: str,
        samples_per_class: int = 500
    ) -> float:
        """생성모델 평가 (생성 + 분류 학습 + 성능 측정)"""
        self.logger.log(f"\n=== Evaluating {model_name} ===")
        
        # 생성
        gen_images, gen_labels = self.generate_with_model(
            model, model_name, samples_per_class
        )
        
        # 원본 + 생성 데이터 병합
        merged_images, merged_labels = merge_datasets(
            self.original_images,
            self.original_labels,
            gen_images,
            gen_labels
        )
        
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
            save_dir=self.exp_dir / "generative" / model_name,
            seed=self.seed
        )
        
        self.results['generative_models'][model_name] = best_acc
        self.logger.log(f"{model_name} accuracy: {best_acc:.2f}%")
        
        return best_acc
    
    def compute_all_fids(self):
        """모든 생성 모델의 FID 계산"""
        self.logger.log("\n=== Computing FID scores ===")
        
        # Real set: CIFAR-10 전체 학습 데이터
        full_train_dataset = self.few_shot.get_full_train_dataset(transform=None)
        real_images, _ = extract_images_and_labels(full_train_dataset)
        
        self.logger.log(f"Real set size: {len(real_images)}")
        
        # 각 생성 모델의 FID 계산
        for model_name in self.results['generative_models'].keys():
            gen_path = GENERATED_DIR / f"{model_name}.pkl"
            
            if not gen_path.exists():
                self.logger.log(f"Warning: {gen_path} not found")
                continue
            
            # 생성 이미지 로드
            with open(gen_path, 'rb') as f:
                data = pickle.load(f)
                fake_images = data['images']
            
            # FID 계산
            fid = compute_fid(real_images, fake_images, batch_size=50)
            
            self.results['fid_scores'][model_name] = fid
            self.logger.log(f"{model_name} FID: {fid:.2f}")
    
    def select_best_model(self) -> str:
        """최적 모델 선정 (분류 성능 기준)"""
        gen_results = self.results['generative_models']
        
        best_model = max(gen_results.items(), key=lambda x: x[1])
        best_name, best_acc = best_model
        
        self.results['best_model'] = {
            'name': best_name,
            'accuracy': best_acc,
            'fid': self.results['fid_scores'].get(best_name, None)
        }
        
        self.logger.log(f"\n=== Best Model ===")
        self.logger.log(f"Model: {best_name}")
        self.logger.log(f"Accuracy: {best_acc:.2f}%")
        self.logger.log(f"FID: {self.results['fid_scores'].get(best_name, 'N/A')}")
        
        return best_name
    
    def run_full_experiment(self):
        """전체 실험 실행"""
        self.logger.log("=" * 60)
        self.logger.log("Objective 1: Optimal Generative Model Search")
        self.logger.log("=" * 60)
        
        # 1. 전통적 증강 준비
        self.prepare_traditional_augmentation(augmentations_per_sample=5)
        
        # 2. Baseline 실험
        self.run_baseline1()
        self.run_baseline2()
        
        # 3. LoRA 모델 학습
        self.train_lora_models()
        
        # 4. 생성 모델 평가
        # Non-finetuned
        self.evaluate_generative_model(
            self.base_diffusion_model, "non_finetuned", samples_per_class=500
        )
        
        # LoRA models
        for rank in LORA_CONFIG["ranks"]:
            for case in ['a', 'b']:
                model_path = CHECKPOINT_DIR / f"lora_case_{case}_rank{rank}.pt"
                
                if model_path.exists():
                    model = self.base_diffusion_model  # 실제로는 LoRA 로드
                    model_name = f"lora_case_{case}_rank{rank}"
                    self.evaluate_generative_model(model, model_name, samples_per_class=500)
        
        # 5. FID 계산
        self.compute_all_fids()
        
        # 6. 최적 모델 선정
        best_model = self.select_best_model()
        
        # 7. 결과 저장
        save_results(self.results, self.exp_dir / "results.json")
        
        # 8. 시각화
        self.plot_results()
        
        self.logger.log("\n=== Experiment completed ===")
        self.logger.close()
        
        return self.results
    
    def plot_results(self):
        """결과 시각화"""
        # 분류 성능 비교
        methods = list(self.results['baselines'].keys()) + \
                  list(self.results['generative_models'].keys())
        accuracies = list(self.results['baselines'].values()) + \
                     list(self.results['generative_models'].values())
        
        plot_comparison_bar(
            methods, accuracies,
            save_path=self.exp_dir / "accuracy_comparison.png",
            title="Classification Accuracy Comparison"
        )
        
        # FID 비교
        if self.results['fid_scores']:
            fid_methods = list(self.results['fid_scores'].keys())
            fid_values = list(self.results['fid_scores'].values())
            
            plot_comparison_bar(
                fid_methods, fid_values,
                save_path=self.exp_dir / "fid_comparison.png",
                title="FID Score Comparison",
                ylabel="FID (lower is better)"
            )


if __name__ == "__main__":
    print("Objective 1 experiment module loaded.")
    print("Note: Requires pretrained diffusion model to run.")