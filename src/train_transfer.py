
"""Train and evaluate a MobileNetV2 brain tumor MRI classifier."""

import argparse
import tensorflow as tf

from src.data_utils import (
    IMG_SIZE,
    ensure_dirs,
    load_datasets,
    save_class_names,
    callbacks,
    plot_history,
    evaluate_model,
)


def build_model(num_classes):
    """Build a MobileNetV2 transfer-learning model."""

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.04, 0.04),
        ],
        name="augmentation",
    )

    # Load the pretrained MobileNetV2 feature extractor.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image")

    x = augmentation(inputs)

    # Equivalent to MobileNetV2 preprocess_input for RGB pixels in [0, 255].
    # Rescaling is serializable in the native .keras model format.
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_rescaling",
    )(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv2_brain_mri",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base


def main():
    parser = argparse.ArgumentParser(
        description="Train a four-class brain tumor MRI classifier."
    )

    parser.add_argument(
        "--train-dir",
        default="data/raw/train_data/train",
        help="Directory containing the training class folders.",
    )
    parser.add_argument(
        "--test-dir",
        default="data/raw/test_data/test",
        help="Directory containing the testing class folders.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)

    args = parser.parse_args()

    ensure_dirs()

    # Load training, validation, and test datasets.
    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.train_dir,
        args.test_dir,
        args.batch_size,
    )

    print("\nDetected classes:", class_names)
    print("Number of classes:", len(class_names))

    if len(class_names) != 4:
        raise ValueError(
            f"Expected 4 classes, but found {len(class_names)}: {class_names}. "
            "Check the folders inside your training directory."
        )

    save_class_names(class_names)

    # Use the native Keras format to avoid legacy H5 serialization issues.
    model_path = "models/mobilenetv2_transfer.keras"

    model, base = build_model(len(class_names))
    model.summary()

    print("\nStage 1: Training the classification head...")

    first_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks(model_path),
    )

    print("\nStage 2: Fine-tuning the final MobileNetV2 layers...")

    base.trainable = True

    # Freeze earlier layers and fine-tune only the final 30 layers.
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen for more stable fine-tuning.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    second_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks(model_path),
    )

    # Combine histories for a single training-curve plot.
    merged_history = type("History", (), {})()
    merged_history.history = {}

    all_keys = set(first_history.history) | set(second_history.history)

    for key in all_keys:
        merged_history.history[key] = (
            first_history.history.get(key, [])
            + second_history.history.get(key, [])
        )

    plot_history(
        merged_history,
        "outputs/mobilenetv2_history.png",
        "MobileNetV2 Transfer Learning",
    )

    # Reload the saved best model to verify that serialization works.
    print("\nLoading the saved model...")
    best_model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    if test_ds is not None:
        print("\nEvaluating on the test dataset...")
        evaluate_model(
            best_model,
            test_ds,
            class_names,
            "mobilenetv2",
        )

    print(f"\nTraining completed. Model saved to: {model_path}")
    print("Class names saved to: models/class_names.json")
    print("Training plot saved to: outputs/mobilenetv2_history.png")


if __name__ == "__main__":
    main()
