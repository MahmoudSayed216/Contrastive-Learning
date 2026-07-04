import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from dataset import MNISTDataset
from model import SiameseRIZZNet


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("triplet_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if main() is re-run in a notebook

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_path}")
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg and device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def df_to_images_labels(df):
    """Turn a raw MNIST dataframe (as loaded inside MNISTDataset) into a flat
    (images, labels) tensor pair, without any triplet sampling. Used for KNN eval."""
    labels = torch.from_numpy(df["label"].to_numpy())
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    images = torch.from_numpy(pixels.reshape(-1, 1, 28, 28))
    return images, labels


# --------------------------------------------------------------------------- #
# Embedding / KNN evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_embeddings(model: nn.Module, images: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    embeddings = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        emb = model(batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0).numpy()


def compute_knn_metrics(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    k: int,
    max_train_samples: int = None,
    seed: int = 42,
) -> dict:
    """Embed train + test sets and score a KNN classifier in embedding space.
    This is a pure evaluation metric, it plays no role in the triplet loss training.

    Since MNIST has 10 classes, precision/recall/f1 are macro-averaged: computed
    per-digit then averaged unweighted across all 10 digits, rather than pooled
    globally (which would just reduce to accuracy for a single-label problem)."""
    if max_train_samples is not None and max_train_samples < train_images.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(train_images.shape[0], size=max_train_samples, replace=False)
        train_images = train_images[idx]
        train_labels = train_labels[idx]

    train_emb = extract_embeddings(model, train_images, device)
    test_emb = extract_embeddings(model, test_images, device)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_emb, train_labels.numpy())

    test_labels_np = test_labels.numpy()
    predictions = knn.predict(test_emb)

    accuracy = accuracy_score(test_labels_np, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_np, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, criterion, device, logger, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)

    for batch_idx, (anchor, positive, negative) in enumerate(loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_every == 0:
            logger.info(
                f"  [Epoch {epoch}/{total_epochs}] Batch {batch_idx + 1}/{n_batches} "
                f"| Batch Triplet Loss: {loss.item():.4f}"
            )

    return running_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_batches = len(loader)

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)
        running_loss += loss.item()

    return running_loss / n_batches


# --------------------------------------------------------------------------- #
# Training orchestration
# --------------------------------------------------------------------------- #

def train(cfg: dict):
    logger = setup_logging(cfg.get("LOG_DIR", "logs"))
    set_seed(cfg.get("SEED", 42))

    device = get_device(cfg.get("DEVICE", "auto"))
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_dataset = MNISTDataset(csv_path="data/mnist_train.csv")
    val_dataset = MNISTDataset(csv_path="data/mnist_test.csv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg.get("NUM_WORKERS", 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg.get("NUM_WORKERS", 0),
        drop_last=True,
    )

    # Flat (image, label) tensors for KNN evaluation, reusing the dataframes
    # already loaded inside the triplet datasets (no extra CSV reads).
    train_images, train_labels = df_to_images_labels(train_dataset.df)
    test_images, test_labels = df_to_images_labels(val_dataset.df)

    model = SiameseRIZZNet().to(device)
    criterion = nn.TripletMarginLoss(margin=cfg.get("MARGIN", 1.0))
    optimizer = optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.get("SCHEDULER_FACTOR", 0.5),
        patience=cfg.get("SCHEDULER_PATIENCE", 3),
        min_lr=cfg.get("SCHEDULER_MIN_LR", 1e-6),
    )

    checkpoint_dir = cfg.get("CHECKPOINT_DIR", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    epochs = cfg["EPOCHS"]
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, logger, epoch, epochs
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        knn_metrics = compute_knn_metrics(
            model,
            train_images,
            train_labels,
            test_images,
            test_labels,
            device,
            k=cfg.get("KNN_K", 5),
            max_train_samples=cfg.get("KNN_MAX_SAMPLES", 5000),
            seed=cfg.get("SEED", 42),
        )
        knn_acc = knn_metrics["accuracy"]

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{epochs} | Train Triplet Loss: {train_loss:.4f} "
            f"| Val Triplet Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )
        logger.info(
            f"  KNN (k={cfg.get('KNN_K', 5)}) -> Acc: {knn_metrics['accuracy']:.4f} "
            f"| Precision (macro): {knn_metrics['precision']:.4f} "
            f"| Recall (macro): {knn_metrics['recall']:.4f} "
            f"| F1 (macro): {knn_metrics['f1']:.4f}"
        )

        # scheduler watches validation triplet loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "knn_metrics": knn_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(f"  -> New best model (val_loss={val_loss:.4f}) saved to {ckpt_path}")

    logger.info("Training complete.")
    logger.info(f"Best validation triplet loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train a Siamese network with Triplet Loss on MNIST.")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()