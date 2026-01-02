# notebooks/01_xai_demo.py

import sys
import os

# Ajoute le répertoire parent (racine du projet) au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.dataloader import get_dataloaders
from src.models import ResNet18
from src.xai import (
    explain_with_lime,
    explain_with_shap,
    compute_saliency,
    explain_with_gradcam,
    generate_counterfactual
)
from src.utils.visualization import show_explanation


def main():
    # 1. Préparation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    # 2. Charger le modèle
    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("checkpoints/resnet18_cifar10.pth", map_location=device))
    model.eval()

    # 3. Charger quelques images de validation
    _, val_loader = get_dataloaders(batch_size=1, num_workers=0)
    data_iter = iter(val_loader)
    image, label = next(data_iter)
    image = image.squeeze(0).to(device)
    true_class = label.item()

    # 4. Prédiction du modèle
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

    print(f"Vraie classe : {true_class}")
    print(f"Prédiction : {predicted_class} (confiance : {confidence:.3f})")

    # 5. Expliquer avec LIME
    print("Calcul de l'explication LIME...")
    lime_mask = explain_with_lime(model, image, predicted_class, device, num_samples=500)
    show_explanation(image, lime_mask, title="LIME")

    # 6. Expliquer avec SHAP
    print("Calcul de l'explication SHAP...")
    shap_attributions = explain_with_shap(model, image, predicted_class, device)
    show_explanation(image, shap_attributions.abs().max(dim=0)[0], title="SHAP")

    # 7. Saliency map
    print("Calcul de la saliency map...")
    saliency_map = compute_saliency(model, image, predicted_class, device)
    show_explanation(image, saliency_map.abs().max(dim=0)[0], title="Saliency")

    # 8. Grad-CAM (choisir une couche conv du modèle)
    print("Calcul de la carte Grad-CAM...")
    target_layer = model.layer4[-1]  # dernière couche conv du dernier étage
    gradcam_map = explain_with_gradcam(model, image, target_layer, predicted_class, device)
    show_explanation(image, gradcam_map, title="Grad-CAM")

    # 9. Contre-exemple (attention : pas réaliste, pédagogique)
    print("Génération d'un contre-exemple...")
    target_class = (predicted_class + 1) % 10  # changer la classe de +1
    cf_image, _ = generate_counterfactual(
        model, image, predicted_class, target_class, device, num_steps=50, step_size=0.01
    )
    show_explanation(image, cf_image, title=f"Contre-exemple (vers classe {target_class})")

    # 10. Affichage
    plt.show()


if __name__ == "__main__":
    main()
