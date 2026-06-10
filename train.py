"""
Train the quality classifier (MobileNetV3-Small, binary good/bad).

HOW TRAINING WORKS
──────────────────
1. Load all images from training_data/good/ (label=1) and training_data/bad/ (label=0).
2. Balance the classes by under-sampling whichever has more images so the
   model doesn't learn to simply always predict the majority class.
3. Split 80% for training, 20% for validation.
4. Fine-tune a MobileNetV3-Small pre-trained on ImageNet — this gives a
   big head-start since the model already understands edges, textures, etc.
5. Replace the final classifier layer with a 2-output layer (good/bad).
6. Train for 15 epochs with Adam optimiser.
7. Evaluate on the validation set and report accuracy.
8. Save as both model.pt (torch) and model.onnx (for inference_worker.py).

Run standalone:   py -3.12 train.py
Or via Train button in labeling_tool.py (runs this as a subprocess).
"""
import os
import random

# Must be set BEFORE importing torch to prevent CUDA DLL crash on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Where to save the trained model files
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

# Training hyper-parameters
IMG_SIZE   = 224   # MobileNetV3 expects 224×224 input
BATCH_SIZE = 16    # images per training step (higher = faster but more RAM)
EPOCHS     = 15    # full passes through the dataset
LR         = 1e-4  # learning rate for Adam (small because we're fine-tuning)


def train(good_dir: str, bad_dir: str, progress_cb=None) -> float:
    """
    Train on images in good_dir and bad_dir.
    progress_cb(epoch, total_epochs, avg_loss) — called after each epoch.
    Returns validation accuracy (0.0 – 1.0).
    Saves model.pt and model.onnx to the project folder.
    """
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    from torch import nn, optim
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    # ── Custom dataset class ──────────────────────────────────────────────────

    class FrameDataset(Dataset):
        """Loads images from disk and applies transforms on demand."""
        def __init__(self, paths, labels, transform):
            self.paths     = paths
            self.labels    = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")   # load as RGB
            return self.transform(img), self.labels[i]

    # ── Gather file lists ─────────────────────────────────────────────────────

    _exts = (".jpg", ".jpeg", ".png")
    good_files = sorted(                          # sort for reproducibility
        os.path.join(good_dir, f)
        for f in os.listdir(good_dir) if f.lower().endswith(_exts)
    )
    bad_files = sorted(
        os.path.join(bad_dir, f)
        for f in os.listdir(bad_dir) if f.lower().endswith(_exts)
    )

    # Under-sample the majority class so both classes are equal size
    n = min(len(good_files), len(bad_files))
    random.shuffle(good_files)
    random.shuffle(bad_files)

    # label 1 = good, label 0 = bad  (must match inference_worker.py)
    files  = good_files[:n] + bad_files[:n]
    labels = [1] * n        + [0] * n

    # Shuffle good and bad together so batches are mixed
    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)

    # ── Train / validation split ──────────────────────────────────────────────

    split       = int(len(files) * 0.8)   # 80% train, 20% val
    train_files = list(files[:split]);  train_lbl = list(labels[:split])
    val_files   = list(files[split:]);  val_lbl   = list(labels[split:])

    # Training augmentation: random flips + slight colour jitter to improve
    # generalisation (the model sees each image in slightly different forms)
    tfm_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    # Validation: no augmentation — just resize and normalise
    tfm_val = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # DataLoaders handle batching and shuffling automatically
    train_dl = DataLoader(
        FrameDataset(train_files, train_lbl, tfm_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_dl = DataLoader(
        FrameDataset(val_files, val_lbl, tfm_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ── Build model ───────────────────────────────────────────────────────────

    # CPU only (CUDA_VISIBLE_DEVICES="" at the top ensures this)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}  |  {n} good + {n} bad  |  {EPOCHS} epochs")

    # Load MobileNetV3-Small with ImageNet weights as a starting point
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Replace the final output layer: ImageNet has 1000 classes, we need 2
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()   # standard loss for classification

    # ── Training loop ─────────────────────────────────────────────────────────

    for epoch in range(1, EPOCHS + 1):
        model.train()   # enable dropout / batch-norm training behaviour
        total_loss = 0.0

        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()              # clear gradients from last step
            loss = criterion(model(imgs), lbls)  # forward pass + compute loss
            loss.backward()                    # backprop — compute gradients
            optimizer.step()                   # update weights
            total_loss += loss.item()

        avg = total_loss / len(train_dl)       # average loss across all batches
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg:.4f}")
        if progress_cb:
            progress_cb(epoch, EPOCHS, avg)    # update GUI progress bar

    # ── Validation ────────────────────────────────────────────────────────────

    model.eval()   # disable dropout for evaluation
    correct = total = 0

    with torch.no_grad():   # no gradient tracking needed for inference
        for imgs, lbls in val_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(dim=1) == lbls).sum().item()
            total   += len(lbls)

    acc = correct / total if total else 0.0
    print(f"Val accuracy: {acc*100:.1f}%")

    # ── Save model ────────────────────────────────────────────────────────────

    # Save full model object (.pt) as a backup
    torch.save(model, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")

    # Export to ONNX format — used by inference_worker.py to avoid DLL issues
    onnx_path = MODEL_PATH.replace(".pt", ".onnx")
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)   # example input for tracing
    model.eval()
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,   # widely supported; opset 17+ requires onnxscript
    )
    print(f"Saved: {onnx_path}")
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
