"""Train a custom CNN from scratch."""
import argparse
import tensorflow as tf
from src.data_utils import (
    IMG_SIZE, ensure_dirs, load_datasets, save_class_names, callbacks,
    plot_history, evaluate_model
)

def build_model(num_classes):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1./255)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="custom_brain_mri_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/raw/Training")
    parser.add_argument("--test-dir", default="data/raw/Testing")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    ensure_dirs()
    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.train_dir, args.test_dir, args.batch_size
    )
    save_class_names(class_names)
    model_path = "models/custom_cnn.h5"
    model = build_model(len(class_names))
    model.summary()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=callbacks(model_path)
    )
    plot_history(history, "outputs/custom_cnn_history.png", "Custom CNN")
    if test_ds is not None:
        best = tf.keras.models.load_model(model_path)
        evaluate_model(best, test_ds, class_names, "custom_cnn")
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    main()
