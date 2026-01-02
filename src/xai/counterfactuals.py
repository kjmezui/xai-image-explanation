import torch
import torch.nn.functional as F


def generate_counterfactual(
        model,
        image,
        original_class,
        target_class,
        device,
        num_steps=100,
        step_size=0.01
):
    """
    Génère une image contre-factuelle simple :
    on part de l'image initiale et on applique de petites modifications
    (gradient ascent vers la clase cible).

    - model : modèle PyTorch
    - image : tenseur (3, H, W) normalisé
    - orginal_class : classe de départ (pour information / affichage)
    - target_class : classe souhaitée
    - device : 'cpu', 'mps' ou 'cuda'
    - num_steps : nombre d'itérations de mise à jour
    - step_size : pas de mise à jour

    Retourne l'image modifiée (3, H, W) et l'historique des probabilités.
    """

    model.eval()

    # On travaulle sur une copie de l'image
    x = image.clone().unsqueeze(0).to(device)
    x.requires_grad = True

    probabilities_history = []

    for _ in range(num_steps):
        # On remet les gradients à zéro
        model.zero_grad()

        # Prédiction actuelle
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probabilities_history.append(probs.detach().cpu())

        # On prend la probabilité de la classe cible
        target_score = probs[0, target_class]

        # On veut augmenter cette probabilité -> gradient ascent
        loss = -target_score # signe moins pour faire une "maximisation"
        loss.backward()

        # Mise à nour de l'image dans la direction du gradient
        with torch.no_grad():
            x = x - step_size * x.grad.sign()
            x = x.clamp(-1.0, 1.0)  # garder l'image dans une plage raisonnable (si normalisée)
            x.requires_grad = True

    # On enlève la dimension batch pour le retour
    x_cf = x.detach().cpu().squeeze(0)  

    return x_cf, probabilities_history  