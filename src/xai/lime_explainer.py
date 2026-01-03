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

# Fonction explain_with_lime corrigée
def explain_with_lime(model, image, class_idx, device, num_samples=1000, top_labels=10):
    model.eval()

    # Dé-normalisation
    image_denorm = image.clone()
    image_denorm = image_denorm * 0.5 + 0.5
    image_np = image_denorm.permute(1, 2, 0).cpu().numpy()

    # Explainer LIME
    explainer = lime_image.LimeImageExplainer()  # ← Maintenant ça marche !

    explanation = explainer.explain_instance(
        image_np,
        classifier_fn=lambda imgs: _predict_proba_numpy(model, imgs, device),
        top_labels=top_labels,  # 10 labels au lieu de 5
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10),
    )

    # Si la classe demandée n'est pas dans les top_labels, prendre la meilleure
    labels = explanation.top_labels
    if class_idx not in labels:
        class_idx = labels[0]

    _, mask = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    return mask