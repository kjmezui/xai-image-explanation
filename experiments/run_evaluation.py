
#!/usr/bin/env python3
# experiments/run_evaluation.py

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import get_dataloaders
from src.models import ResNet18
from src.xai import (
    explain_with_lime,
    explain_with_shap,
    compute_saliency,
    explain_with_gradcam,
)
from src.evaluation.metrics import deletion_score
from src.evaluation.sanity_checks import (
    random_baseline_correlation,
    weight_randomization_test,
)
from src.utils.logger import get_logger
from src.utils.visualization import denormalize_image


def saliency_wrapper(model, img, cls, device):
    """Wrapper pour utiliser compute_saliency avec sanity checks."""
    return compute_saliency(model, img, cls, device).abs().max(dim=0)[0]


def lime_wrapper(model, img, cls, device):
    mask = explain_with_lime(model, img, cls, device, num_samples=200, top_labels=10)
    return torch.tensor(mask, dtype=torch.float32)


def main(num_images=10, device=None):
    logger = get_logger("evaluation")
    
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Utilisation du device : {device}")

    # 1. Charger le modèle
    model = ResNet18(num_classes=10).to(device)
    checkpoint_path = "checkpoints/resnet18_cifar10.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info("Modèle chargé")

    # 2. Charger quelques images de validation
    _, val_loader = get_dataloaders(batch_size=1, num_workers=0)
    
    results = {
        "lime": {"deletion_auc": [], "random_corr": [], "weight_corr": []},
        "shap": {"deletion_auc": [], "random_corr": [], "weight_corr": []},
        "saliency": {"deletion_auc": [], "random_corr": [], "weight_corr": []},
        "gradcam": {"deletion_auc": [], "random_corr": [], "weight_corr": []},
    }


    # 3. Évaluer sur num_images
    for i in range(num_images):
        image, true_label = next(iter(val_loader))
        image = image.squeeze(0).to(device)
        true_label = true_label.item()

        with torch.no_grad():
            logits = model(image.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()

        logger.info(f"Image {i+1}: vraie={true_label}, prédite={pred_class}")

        target_layer = model.layer4[-1]

        # LIME (AUC + random corr seulement)
        lime_map = lime_wrapper(model, image, pred_class, device)
        results["lime"]["deletion_auc"].append(deletion_score(model, image, lime_map, pred_class, device))
        results["lime"]["random_corr"].append(random_baseline_correlation(lime_map))
        results["lime"]["weight_corr"].append(np.nan)

        # SHAP
        shap_map = explain_with_shap(model, image, pred_class, device).abs().max(dim=0)[0]
        results["shap"]["deletion_auc"].append(deletion_score(model, image, shap_map, pred_class, device))
        results["shap"]["random_corr"].append(random_baseline_correlation(shap_map))
        results["shap"]["weight_corr"].append(
            weight_randomization_test(model, lambda img, cls: explain_with_shap(model, img, cls, device).abs().max(dim=0)[0], image, pred_class, device)
            )

        # Saliency
        saliency_map = compute_saliency(model, image, pred_class, device).abs().max(dim=0)[0]
        results["saliency"]["deletion_auc"].append(deletion_score(model, image, saliency_map, pred_class, device))
        results["saliency"]["random_corr"].append(random_baseline_correlation(saliency_map))
        results["saliency"]["weight_corr"].append(
            weight_randomization_test(model, lambda img, cls: compute_saliency(model, img, cls, device).abs().max(dim=0)[0], image, pred_class, device)
            )

        # Grad-CAM
        gradcam_map = explain_with_gradcam(model, image, target_layer, pred_class, device)
        results["gradcam"]["deletion_auc"].append(deletion_score(model, image, gradcam_map, pred_class, device))
        results["gradcam"]["random_corr"].append(random_baseline_correlation(gradcam_map))
        results["gradcam"]["weight_corr"].append(
            weight_randomization_test(model, lambda img, cls: explain_with_gradcam(model, img, target_layer, cls, device), image, pred_class, device)
            )

    # 4. Afficher les résultats
    logger.info("\n=== RÉSULTATS ÉVALUATION (moyennes sur %d images) ===", num_images)
    
    for method, metrics in results.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  AUC Deletion : {np.mean(metrics['deletion_auc']):.3f} ± {np.std(metrics['deletion_auc']):.3f}")
        logger.info(f"  Corr. aléatoire : {np.mean(metrics['random_corr']):.3f} ± {np.std(metrics['random_corr']):.3f}")
        logger.info(f"  Corr. poids rand. : {np.mean(metrics['weight_corr']):.3f} ± {np.std(metrics['weight_corr']):.3f}")

    # 5. Sauvegarder les résultats
    np.savez("evaluation_results.npz", results)
    logger.info("Résultats sauvés dans 'evaluation_results.npz'")


if __name__ == "__main__":
    main(num_images=10)
