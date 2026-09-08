import os
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR, "dataset", "brain_tumor_dataset", "Training"
)

TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "brain_tumor_dataset", "Testing"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR, "brain_tumor_cnn_4class.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123


# Check dataset folders
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Testing dataset not found: {TEST_DIR}"
    )

os.makedirs(MODEL_DIR, exist_ok=True)


# Load training dataset
print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)


# Load testing dataset
print("\nLoading testing dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)


# Get class names
class_names = train_ds.class_names

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {len(class_names)}")

if len(class_names) != 4:
    raise ValueError(
        f"Expected 4 classes, but found {len(class_names)}."
    )


# Improve dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# Build CNN model
print("\nBuilding CNN model...")

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(4, activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Display model
print("\nModel architecture:\n")

model.summary()


# Train model
print("\nStarting training...")

print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE}")

print("\n")

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)


# Evaluate model
print("\nEvaluating model on testing dataset...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)


# Display results
print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("==============================")


# Save model
print("\nSaving trained model...")

model.save(MODEL_PATH)

print("\nModel saved successfully!")

print(f"Location: {MODEL_PATH}")