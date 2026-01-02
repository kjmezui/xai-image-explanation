# src/xai/gradcam.py

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def explain_with_gradcam(
    model, image, target_layer, class_idx, device
):
    """
    Explique une image avec Grad-CAM.
    """
    model.eval()

    # On ajoute la dimension batch
    input_tensor = image.unsqueeze(0).to(device)

    # GradCAM v2+ : plus d'arg 'use_cuda', on passe juste model et target_layers
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        # use_cuda supprimé (interne à la lib)
    )

    # Cible : classe pour laquelle on veut l'explication
    targets = [ClassifierOutputTarget(class_idx)]

    # Calcul de la carte Grad-CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )

    # 'grayscale_cam' a la forme (N, H, W). On prend le premier (N=1)
    cam_map = grayscale_cam[0, :]

    # On renvoie la carte en tant que tenseur PyTorch CPU
    return torch.tensor(cam_map)
