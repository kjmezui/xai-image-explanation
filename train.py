#!/usr/bin/env python3
# train.py

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Ajoute la racine du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import get_dataloaders
from src.models import ResNet18
from src.utils.logger import get_logger
from src.utils.config import load_config_from_yaml


def main():
    logger = get_logger()
    
    # Charger la config
    cfg = load_config_from_yaml("config.yaml")
    logger.info(f"Config chargée : {cfg.training}")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Utilisation du device : {device}")

    # Data
    train_loader, _ = get_dataloaders(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    logger.info(f"DataLoader prêt : {len(train_loader.dataset)} images")

    # Model
    model = ResNet18(num_classes=10).to(device)
    logger.info("Modèle ResNet18 créé")

    # Loss + optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Entraînement
    model.train()
    for epoch in range(cfg.training.num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch+1} terminée, Loss moyenne: {total_loss/len(train_loader):.4f}")

    # CRÉER LE DOSSIER checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # Sauvegarde
    torch.save(model.state_dict(), cfg.training.checkpoint_path)
    logger.info(f"Modèle sauvegardé : {cfg.training.checkpoint_path}")


if __name__ == "__main__":
    main()

