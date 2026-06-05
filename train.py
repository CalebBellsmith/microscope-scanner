"""
Fine-tune MobileNetV3-Small on labeled slide images.
Run: python train.py
Output: model.pt (used by ml_inference.py)
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
CLASSES = ["good", "watermark", "blotch", "vertical_scratch", "debris"]
IMG_SIZE = 224
BATCH = 16
EPOCHS = 10
LR = 1e-4


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No training data found at {DATA_DIR}. Run labeling_tool.py first.")
        sys.exit(1)

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(DATA_DIR, transform=transform_train)
    n_val = max(1, len(full_ds) // 5)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, {n_train} train / {n_val} val samples")

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={train_loss/len(train_loader):.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, MODEL_PATH)
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    print(f"Training complete. Best val acc: {best_val_acc:.3f}. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
