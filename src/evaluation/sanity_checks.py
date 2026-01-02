# src/evaluation/sanity_checks.py

import torch
import numpy as np
from typing import Callable


def random_baseline_correlation(attr_map: torch.Tensor) -> float:
    """
    Compare une carte d'attribution à une carte aléatoire
    via la corrélation de Pearson.
    Idée : si la corrélation est très proche de 1,
    la carte n'apporte pas plus d'info qu'un bruit aléatoire.
    """
    attr = attr_map.detach().cpu().view(-1).numpy()
    random_map = np.random.rand(attr.size)

    if np.std(attr) < 1e-8:
        return 0.0

    corr = np.corrcoef(attr, random_map)[0, 1]
    return float(corr)


def weight_randomization_test(
    model,
    explain_func: Callable[[torch.Tensor, int], torch.Tensor],
    image: torch.Tensor,
    class_idx: int,
    device,
) -> float:
    """
    Test de randomisation des poids :
    - on calcule une carte d'attribution avec les poids appris,
    - on randomise les poids du modèle,
    - on recalcule une carte,
    - on mesure la similarité (corrélation).

    Idée : une méthode saine doit donner des cartes très différentes
    lorsque les poids sont randomisés. [web:99]
    """
    model.eval()
    image = image.to(device)

    # Carte d'attribution avec poids appris
    attr_orig = explain_func(image, class_idx).detach().cpu().view(-1).numpy()

    # On sauvegarde les poids
    state_dict = model.state_dict()

    # On randomise les poids
    for param in model.parameters():
        if param.requires_grad:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)

    # Carte d'attribution avec poids randomisés
    attr_rand = explain_func(image, class_idx).detach().cpu().view(-1).numpy()

    # On restaure les poids
    model.load_state_dict(state_dict)

    # Corrélation entre les deux cartes
    if np.std(attr_orig) < 1e-8 or np.std(attr_rand) < 1e-8:
        return 0.0

    corr = np.corrcoef(attr_orig, attr_rand)[0, 1]
    return float(corr)
