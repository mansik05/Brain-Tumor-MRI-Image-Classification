"""Evaluate an already-trained model on a labeled image directory."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from src.data_utils import IMG_SIZE, evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-dir", default="data/raw/Testing")
    args = parser.parse_args()
    with open("models/class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}. Train it first.")
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.test_dir, image_size=IMG_SIZE, batch_size=32, shuffle=False,
        class_names=class_names, label_mode="int"
    ).cache().prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model(args.model)
    prefix = Path(args.model).stem
    evaluate_model(model, dataset, class_names, prefix)

if __name__ == "__main__":
    main()
