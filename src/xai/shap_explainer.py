import torch
import captum.attr as attr


def explain_with_shap(model, image, class_idx, device):
    """
    Explique une image avec GradientShap (via Captum).
    - model : modèle Pytorch
    - image : tenseur (3, H, W) normalisé
    - class_idx : indice de la classe cible
    - device : 'cpu', 'mps' ou 'cuda'

    Retourne un tenseur (3, H, W) d'attributions (importances par canal/pixel).
    """
    model.eval()

    # On s'assure que l'image est sur le bon device et a un batch dimension (1, 3, H,
    # W)

    input_tensor = image.unsqueeze(0).to(device)

    # On doit définir une baseline. Pour les images, fréquemment :
    # - une image noire
    # - ou une image moyenne
    baseline = torch.zeros_like(input_tensor).to(device)

    # Création de l'objet GradientShap
    gradient_shap = attr.GradientShap(model)

    # Calcul des attributions
    attributions = gradient_shap.attribute(
        input_tensor,
        baselines=baseline,
        target=class_idx
    )

    # On enlève la dimension batch -> (3, H, W)
    attributions = attributions.squeeze(0).detach().cpu()

    return attributions