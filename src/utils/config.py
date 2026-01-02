# src/utils/config.py

import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 2
    dataset_name: str = "cifar10"
    checkpoint_path: str = "checkpoints/resnet18_cifar10.pth"


@dataclass
class XAIConfig:
    num_lime_samples: int = 500
    gradcam_layer: str = "layer4"  # nom logique, résolu dans le code
    saliency_target: str = "predicted"  # ou "true"


@dataclass
class ProjectConfig:
    training: TrainingConfig
    xai: XAIConfig


def load_config_from_yaml(path: str) -> ProjectConfig:
    """
    Charge un fichier YAML et le convertit en objets dataclasses.
    """
    with open(path, "r") as f:
        cfg_dict: Dict[str, Any] = yaml.safe_load(f)

    training_cfg = TrainingConfig(**cfg_dict.get("training", {}))
    xai_cfg = XAIConfig(**cfg_dict.get("xai", {}))

    return ProjectConfig(training=training_cfg, xai=xai_cfg)
