XAI Image Classification Pipeline
[![Python](https://img.shields.io/badge/Python-3.8%2![PyTorch](https://img.shields.io/badge/PyTorch-2.0%![License](https://img.shields.io/badge/License-MIT-greenProjet pédagogique complet d'IA explicable : classification d'images CIFAR-10 avec ResNet-18 + techniques XAI avancées (LIME, SHAP, Saliency, Grad-CAM, contre-factuels) + évaluation quantitative.

🎯 Objectifs pédagogiques
Maîtriser PyTorch (modèles, DataLoader, entraînement)

Implémenter et comparer 5 méthodes d'explicabilité

Évaluer quantitativement les explications (AUC deletion, sanity checks)

Structurer un projet ML professionnel (src/, notebooks/, config YAML)

📁 Structure du projet

xai-image-classification/
├── src/
│   ├── data/          # Datasets + transformations
│   ├── models/        # ResNet18 + CNN simple
│   ├── xai/           # LIME, SHAP, Saliency, Grad-CAM, Counterfactuals
│   ├── evaluation/    # Métriques + sanity checks
│   └── utils/         # Config, logger, visualisation
├── notebooks/         # Scripts de démo interactifs
├── checkpoints/       # Modèles entraînés (.pth)
├── logs/             # Logs d'entraînement
├── data/             # CIFAR-10 (téléchargé auto)
├── config.yaml       # Paramètres du projet
├── train.py          # Entraînement
├── requirements.txt  # Dépendances
└── README.md         # Ce fichier

🚀 Installation rapide (Mac M1/M2/M3)

1. Cloner et créer l'environnement

git clone <repo-url>
cd xai-image-classification
python -m venv myvenv
source myvenv/bin/activate  # Linux/Mac
# myvenv\Scripts\activate  # Windows

2. Installer les dépendances

bash
pip install torch torchvision torchaudio
pip install captum lime grad-cam matplotlib pyyaml
Note Mac M1 : PyTorch utilise MPS automatiquement (torch.device("mps")).

3. Vérifier l'installation

bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"

🎮 Utilisation
Étape 1 : Entraîner le modèle

bash
mkdir -p checkpoints logs data
python train.py
✅ Crée checkpoints/resnet18_cifar10.pth

Étape 2 : Tester les explications XAI

bash
python notebooks/01_xai_demo.py
✅ Affiche LIME + SHAP + Saliency + Grad-CAM + contre-factuel sur une image CIFAR-10.

Étape 3 : Évaluer quantitativement (optionnel)

bash
python experiments/run_evaluation.py
✅ Calcule AUC deletion + sanity checks pour comparer les méthodes.

📊 Résultats attendus
Le script 01_xai_demo.py génère :

text
Vraie classe : 3 (chat)
Prédiction : 3 (confiance : 0.92)
Utilisation du device : mps
[Figures matplotlib : image originale + 5 heatmaps XAI]
🔧 Configuration (config.yaml)
text
training:
  batch_size: 128
  num_epochs: 10
  learning_rate: 0.001
  num_workers: 2
  checkpoint_path: "checkpoints/resnet18_cifar10.pth"

xai:
  num_lime_samples: 500
  gradcam_layer: "layer4"
📈 Métriques d'explicabilité implémentées
Méthode Type    Métrique    Interprétation
Deletion AUC    Quantitative    0.2-0.6 Plus faible = meilleure
Random corr.    Sanity check    ~0  Faible = saine
Weight corr.    Sanity check    <0.3    Faible = dépend du modèle
🛠️ Dossiers générés automatiquement
text
checkpoints/    # resnet18_cifar10.pth
logs/           # training.log
data/           # CIFAR-10 (~170MB)
🔬 Méthodes XAI implémentées
LIME : Explications locales agnostiques au modèle

GradientSHAP (Captum) : Attributions Shapley par gradients

Saliency Maps : Gradients simples par pixel

Grad-CAM : Heatmaps sur feature maps (couche layer4[-1])

Counterfactuals : Perturbations minimales changeant la prédiction

📚 Pour aller plus loin

Notebooks supplémentaires

text
notebooks/
├── 01_xai_demo.py           # Démo complète
├── 02_evaluation.py         # Métriques quantitatives
└── 03_custom_dataset.py     # Adapter à tes images
Extensions possibles

 Support MNIST/FashionMNIST

 Score-CAM, Ablation-CAM

 Integrated Gradients (Captum)

 ROAR (Remove Order Agnostic Removal)

❗ Dépannage

Problème    Solution
No module named 'yaml'  pip install pyyaml
MPS non disponible  Utilise device="cpu"
checkpoints/... not found   Lance python train.py d'abord
Erreurs Captum  Vérifie pip install captum

🤝 Contribution

Fork le repo

Crée une branche feat/nouvelle-methode-xai

Ajoute tes tests dans notebooks/

Push et Pull Request !

📄 Licence

MIT License - voir LICENSE
