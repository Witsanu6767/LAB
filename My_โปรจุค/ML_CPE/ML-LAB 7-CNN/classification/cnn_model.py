import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Normalize 0-255 to 0-1 inside the model, so inference code never
    # has to remember to do it
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Light augmentation, only active during training.
    # This is what keeps a small CNN from memorising the training set.
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)

    # 3. Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 4. Dense Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=80, batch_size=2):

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=30, min_lr=1e-5
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "cnn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'cnn_model.keras')}")

    return model, history


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)