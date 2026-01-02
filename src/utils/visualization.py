# src/utils/visualization.py

import torch
import matplotlib.pyplot as plt
import numpy as np


def show_explanation(image, explanation, title="Explanation"):
    """
    Affiche une image et sa carte d'explication.
    Gère les tenseurs PyTorch (C,H,W) et NumPy (H,W,C).
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # === IMAGE ORIGINALE ===
    # Dé-normaliser pour visualisation (si normalisée avec mean=0.5, std=0.5)
    if isinstance(image, torch.Tensor):
        img_vis = image.clone()
        img_vis = img_vis * 0.5 + 0.5  # inverse Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        img_vis = torch.clamp(img_vis, 0, 1)

        # Conversion en NumPy (H,W,C)
        if img_vis.shape[0] == 3:  # C,H,W -> H,W,C
            img_vis = img_vis.permute(1, 2, 0).cpu().numpy()
        else:
            img_vis = img_vis.cpu().numpy()
    else:
        img_vis = np.array(image)

    ax[0].imshow(img_vis)
    ax[0].set_title("Image originale")
    ax[0].axis("off")

    # === CARTE D'EXPLICATION ===
    if isinstance(explanation, torch.Tensor):
        expl_vis = explanation.clone()
        expl_vis = torch.abs(expl_vis)  # valeurs absolues
        expl_vis = torch.clamp(expl_vis, 0, 1)

        # Conversion en NumPy
        if expl_vis.dim() == 3 and expl_vis.shape[0] == 3:  # C,H,W -> H,W,C
            expl_vis = expl_vis.permute(1, 2, 0).cpu().numpy()
        else:  # (H,W) ou autre
            expl_vis = expl_vis.squeeze().cpu().numpy()
    else:
        expl_vis = np.abs(np.array(explanation))
        expl_vis = np.clip(expl_vis, 0, 1)

    # Affichage avec colormap 'jet'
    im = ax[1].imshow(expl_vis, cmap='jet')
    ax[1].set_title(title)
    ax[1].axis("off")

    # Barre de couleur
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Dé-normalise une image CIFAR-10 pour visualisation.
    """
    return torch.clamp(image * 0.5 + 0.5, 0, 1)

