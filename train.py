import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as vision_transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST digit CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", default="mnist_cnn.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = vision_transforms.Compose([
        vision_transforms.RandomAffine(
            degrees=12,
            translate=(0.12, 0.12),
            scale=(0.85, 1.15),
        ),
        vision_transforms.RandomApply([
            vision_transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        ], p=0.15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {running_loss / len(train_loader):.4f} - "
            f"test accuracy: {accuracy:.2f}%"
        )

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
