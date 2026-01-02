import torch
import captum.attr as attr


def compute_saliency(model, image, class_idx, device):
    """
    Calcule une saliency map simple.
    - model : modèle PyTorch
    - image : tenseur (3, H, W) normalisé
    - class_idx : classe cible
    - device : 'cpu', 'mps' ou 'cuda'

    Retourne un tenseur (3, H, W) d'attributions.
    On pourra ensuite prendre la norme sur les canaux pour une carte 2D.
    """
    model.eval()

    # On prépare l'entrée avec un batch de taille 1
    input_tensor = image.unsqueeze(0).to(device)
    input_tensor.requires_grad = True     # on veut les gradients par rapport à l'entrée

    # Création de l'objet Saliency
    saliency = attr.Saliency(model)

    # Calcul des attributions
    attributions = saliency.attribute(
        input_tensor,
        target=class_idx
    )

    # On enlève la dimension batch et on ramène sur CPU
    attributions = attributions.squeeze(0).detach().cpu()

    return attributions