# src/evaluation/metrics.py

import torch
import torch.nn.functional as F
from typing import Callable


def normalize_map(attr_map: torch.Tensor) -> torch.Tensor:
    """
    Normalise une carte d'attribution (H, W) entre 0 et 1.
    """
    m = attr_map.min()
    M = attr_map.max()
    if (M - m) < 1e-8:
        return torch.zeros_like(attr_map)
    return (attr_map - m) / (M - m)


def deletion_score(
    model,
    image: torch.Tensor,
    attr_map: torch.Tensor,
    class_idx: int,
    device,
    steps: int = 20,
) -> float:
    """
    Implémente un "deletion test" simple :
    - on enlève progressivement (en steps) les pixels les plus importants
      selon la carte attr_map,
    - on mesure la probabilité de la classe cible à chaque étape,
    - plus la probabilité chute vite, plus l'explication est "cohérente".

    Retourne l'aire sous la courbe (AUC) estimée numériquement.
    """
    model.eval()

    # image : (3, H, W), attr_map : (H, W)
    attr = normalize_map(attr_map.detach().cpu())
    H, W = attr.shape

    # Indices triés par importance décroissante
    flat = attr.view(-1)
    sorted_indices = torch.argsort(flat, descending=True)

    # On crée une copie de l'image à modifier
    x = image.clone().to(device).unsqueeze(0)

    # Probabilité initiale
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        base_prob = probs[0, class_idx].item()

    # Nombre de pixels à supprimer à chaque étape
    num_pixels = H * W
    pixels_per_step = num_pixels // steps

    probs_list = [base_prob]

    for step in range(1, steps + 1):
        # On calcule quels indices supprimer à ce step
        start = (step - 1) * pixels_per_step
        end = step * pixels_per_step
        current_indices = sorted_indices[start:end]

        # On met ces pixels à zéro (sur les 3 canaux)
        for idx in current_indices:
            h = idx // W
            w = idx % W
            x[0, :, h, w] = 0.0

        # On recalcule la probabilité pour la classe cible
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs[0, class_idx].item())

    # Approximation de l'AUC par la méthode des trapèzes
    import numpy as np

    xs = np.linspace(0.0, 1.0, len(probs_list))
    ys = np.array(probs_list)
    auc = float(np.trapz(ys, xs))

    return auc
