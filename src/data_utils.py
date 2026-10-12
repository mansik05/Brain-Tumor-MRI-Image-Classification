"""Shared data loading, plotting, and evaluation helpers."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix,
)
import pandas as pd
import seaborn as sns

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EXPECTED_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

def _class_folders(folder):
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {p.resolve()}")
    return sorted(x.name for x in p.iterdir() if x.is_dir())

def _validate_classes(train_dir, test_dir):
    train_classes = _class_folders(train_dir)
    missing_train = sorted(set(EXPECTED_CLASSES) - set(train_classes))
    extra_train = sorted(set(train_classes) - set(EXPECTED_CLASSES))
    if missing_train or extra_train:
        raise ValueError(
            f"Training folder must contain exactly these four class folders: {EXPECTED_CLASSES}. "
            f"Found: {train_classes}. Missing: {missing_train}; unexpected: {extra_train}. "
            "Restore the missing training folders from the original dataset. Do not copy images "
            "from the test set into training."
        )
    if test_dir:
        test_classes = _class_folders(test_dir)
        missing_test = sorted(set(EXPECTED_CLASSES) - set(test_classes))
        extra_test = sorted(set(test_classes) - set(EXPECTED_CLASSES))
        if missing_test or extra_test:
            raise ValueError(
                f"Testing folder must contain exactly these four class folders: {EXPECTED_CLASSES}. "
                f"Found: {test_classes}. Missing: {missing_test}; unexpected: {extra_test}."
            )
    return EXPECTED_CLASSES

def load_datasets(train_dir, test_dir=None, batch_size=BATCH_SIZE):
    """Load images from class subfolders; validation is split from training only."""
    class_names = _validate_classes(train_dir, test_dir)
    train_path = Path(train_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=batch_size, label_mode="int",
        class_names=class_names,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_path, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=batch_size, label_mode="int",
        class_names=class_names,
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = None
    if test_dir:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=IMG_SIZE, batch_size=batch_size,
            shuffle=False, label_mode="int", class_names=class_names,
        ).cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names

def save_class_names(class_names, path="models/class_names.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, indent=2)

def callbacks(model_path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

def plot_history(history, output_path, title):
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.get("accuracy", []), label="Training")
    axes[0].plot(hist.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(hist.get("loss", []), label="Training")
    axes[1].plot(hist.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def evaluate_model(model, dataset, class_names, prefix):
    y_true, y_prob = [], []
    for images, labels in dataset:
        y_prob.append(model.predict(images, verbose=0))
        y_true.extend(labels.numpy().tolist())
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "num_test_images": int(len(y_true)),
        "classes": list(class_names),
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(f"outputs/{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        f"outputs/{prefix}_classification_report.csv", index=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(max(7, len(class_names)), max(5, len(class_names) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{prefix}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_confusion_matrix.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics
