import torch
#print(torch.__version__)
#print(torch.backends.mps.is_available())
from lime import lime_image
from skimage.segmentation import slic

def _predict_proba_numpy(model, images_numpy, device):
    """
    Fonction intermédiaire pour LIME.
    LIME travaille avec des images Numpy (H, W, C) dans [0, 255] ou [0, 1].
    Ici, 'images_numpy' a la forme (N, H, W, C).
    On la convertit en tenseur PyTorch, on passe dans le modèle,
    puis on retourne les probabilités sous forme Numpy (N, num_classes).
    """
    model.eval()

    # Conversion vers tenseur PyTorch (N, C, H, W)
    images_tensor = torch.from_numpy(images_numpy).permute(0, 3, 1, 2).float()
    images_tensor = images_tensor.to(device)

    # Pas de gradient
    with torch.no_grad():
        logits = model(images_tensor)            # [N, num_classes]
        probs = torch.softmax(logits, dim=1)     # probabilités
    
    # Retour en Numpy pour LIME
    return probs.cpu().numpy()

def explain_with_lime(model, image, class_idx, device, num_samples=1000):
    """
    Explique une image avec LIME.
    - model : modèle PyTorch (ResNet18, etc.)
    - image : tenseur (3, H, W) normalisé
    - class_idx : indice de la classe à expliquer
    - device : 'cpu', 'mps' ou 'cuda'

    Retourne un masque (H, W) où les pixels importants sont marqués à 1
    (ou valeurs proches de 1) et le reste à 0.
    """

    model.eval()

    # LIME attend une image Numpy (H, W, C) dans l'espace "normal" (0-1, 
    # pas normalisée).
    # Ici, nos images CIFAR-10 sont normalisées avec mean=0.5, std=0.5,
    # donc on "dé-normalise" simplement.
    image_denorm = image.clone()
    image_denorm = image_denorm * 0.5 + 0.5     # inverse de Normalize((0.5), (0.5))

    # Conversion en Numpy (H, W, C)
    image_np = image_denorm.permute(1, 2, 0).cpu().numpy()

    # Création de l'explainer LIME pour les images
    explainer = lime_image.LimeImageExplainer()

    # LIMe va générer autour de cette image des perturbations et interroger notre
    #  modèle.
    explanation = explainer.explain_instance(
        image_np,
        classifier_fn=lambda imgs: _predict_proba_numpy(model, imgs, device),
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10)
    )

    # On récupère un masque pour la classe qui nous intéresse.
    # 'mask' a la même taille que l'image (H, W), avec des "superpixels" importants.
    _, mask = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return mask
