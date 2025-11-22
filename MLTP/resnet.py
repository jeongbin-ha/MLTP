"""
분류 모델 정의 (직접 구현)
CIFAR-10에 최적화된 ResNet-18
- 첫 conv: 3x3, stride=1, padding=1
- MaxPool 제거
- 최종 FC: num_classes
"""

from typing import Type, Callable, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 유틸리티 conv 함수
# ---------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False
    )


# ---------------------------
# BasicBlock (ResNet-18/34)
# ---------------------------
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # conv-bn-relu × 2
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------------
# ResNet 본체
# ---------------------------
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],                  # e.g., [2, 2, 2, 2] for ResNet-18
        num_classes: int = 10,
        zero_init_residual: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64

        # *** CIFAR-10 설정: conv1=3x3/1, MaxPool 제거 ***
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # 제거와 동일

        # 4개 스테이지
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)  # 32x32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 8x8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 4x4

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 가중치 초기화(Kaiming: torchvision 구현과 동일한 스타일)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 옵션: 마지막 BN의 gamma=0으로 residual branch 0부터 시작
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        # downsample이 필요한 경우(채널/stride가 바뀌는 첫 블록)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # CIFAR-10: maxpool 없음
        x = self.maxpool(x)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------
# ResNet-18 생성 함수 (CIFAR-10 설정)
# ---------------------------
def get_resnet18_cifar10(num_classes: int = 10) -> ResNet:
    """
    CIFAR-10용 ResNet-18 (직접 구현)
    - conv1: 3x3 / stride=1 / padding=1
    - MaxPool 제거
    - layer 설정: [2, 2, 2, 2]
    """
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        zero_init_residual=False,
    )
    return model


# ---------------------------
# 간단한 동작 테스트
# ---------------------------
if __name__ == "__main__":
    model = get_resnet18_cifar10(num_classes=10)
    print("Custom ResNet-18(CIFAR-10) 생성 완료")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   전체 파라미터 수: {total_params:,} (학습 가능: {trainable_params:,})")

    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"   입력 크기: {x.shape} → 출력 크기: {y.shape}")  # torch.Size([1, 10])