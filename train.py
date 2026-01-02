# train.py (exemple simplifié, à adapter)

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import get_dataloaders
from src.models import ResNet18


def main():
    # 1. Préparation données
    train_loader, val_loader = get_dataloaders(
        dataset_name="cifar10",
        batch_size=128,
        num_workers=2,
        pin_memory=True
    )

    # 2. Préparation modèle
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNet18(num_classes=10).to(device)

    # 3. Perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Boucle d'entraînement très simple (1 epoch pour l’exemple)
    for epoch in range(1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")

    # 5. Sauvegarde du modèle
    torch.save(model.state_dict(), "checkpoints/resnet18_cifar10.pth")


if __name__ == "__main__":
    main()
