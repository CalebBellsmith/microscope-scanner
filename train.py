"""
Fine-tune MobileNetV3-Small binary classifier: good (1) vs bad (0).

Usage — from labeling_tool.py GUI (Train button), or standalone:
    py -3.12 train.py

Output: model.pt  (loaded by ml_inference.py)
"""
import os
import random

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 15
LR         = 1e-4


def train(good_dir: str, bad_dir: str, progress_cb=None) -> float:
    """
    Train on images in good_dir (label=1) and bad_dir (label=0).
    progress_cb(epoch, total_epochs, avg_loss) called after each epoch.
    Returns validation accuracy (0.0 – 1.0).
    Saves model to model.pt.
    """
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    from torch import nn, optim
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    # ── Dataset ───────────────────────────────────────────────────────────────

    class FrameDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths     = paths
            self.labels    = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), self.labels[i]

    # ── Gather files ──────────────────────────────────────────────────────────

    _exts = (".jpg", ".jpeg", ".png")
    good_files = sorted(
        os.path.join(good_dir, f)
        for f in os.listdir(good_dir) if f.lower().endswith(_exts)
    )
    bad_files = sorted(
        os.path.join(bad_dir, f)
        for f in os.listdir(bad_dir)  if f.lower().endswith(_exts)
    )

    # Balance classes by undersampling the majority
    n = min(len(good_files), len(bad_files))
    random.shuffle(good_files)
    random.shuffle(bad_files)
    files  = good_files[:n] + bad_files[:n]
    labels = [1] * n        + [0] * n

    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)

    # ── Train / val split 80 / 20 ─────────────────────────────────────────────

    split       = int(len(files) * 0.8)
    train_files = list(files[:split]);  train_lbl = list(labels[:split])
    val_files   = list(files[split:]);  val_lbl   = list(labels[split:])

    tfm_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_val = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dl = DataLoader(
        FrameDataset(train_files, train_lbl, tfm_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_dl = DataLoader(
        FrameDataset(val_files, val_lbl, tfm_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Training on: {device}  |  {n} good + {n} bad  |  {EPOCHS} epochs")

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg:.4f}")
        if progress_cb:
            progress_cb(epoch, EPOCHS, avg)

    # ── Validation ────────────────────────────────────────────────────────────

    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(dim=1) == lbls).sum().item()
            total   += len(lbls)
    acc = correct / total if total else 0.0
    print(f"Val accuracy: {acc*100:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────────

    # Save full model object so ml_inference.py can torch.load() it directly
    torch.save(model, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")
    return acc


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
    good = os.path.join(base, "good")
    bad  = os.path.join(base, "bad")
    if not os.path.isdir(good) or not os.path.isdir(bad):
        print("Run labeling_tool.py first to build training_data/good/ and bad/")
    else:
        train(good, bad)
