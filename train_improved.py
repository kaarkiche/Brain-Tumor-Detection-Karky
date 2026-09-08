import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "brain_tumor_dataset",
    "Training"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "brain_tumor_dataset",
    "Testing"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_tumor_efficientnet.keras"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Testing folder not found:\n{TEST_DIR}"
    )

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES
)


# ============================================================
# 5. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES
)


# ============================================================
# 6. LOAD TEST DATA
# ============================================================

print("\nLoading testing dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES
)


print("\nClasses:")

for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")


# ============================================================
# 7. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")


# ============================================================
# 9. BUILD EFFICIENTNET MODEL
# ============================================================

print("\nBuilding EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    4,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

print("\nModel architecture:\n")

model.summary()


# ============================================================
# 12. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n======================================")
print("STARTING EFFICIENTNET TRAINING")
print("======================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

print("\n======================================")
print("FINAL TEST EVALUATION")
print("======================================")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy * 100:.2f}%")


# ============================================================
# 15. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_labels
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n                 Predicted")
print("             ", CLASS_NAMES)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:12} {row}"
    )


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY")
print("======================================")

print(
    f"Model: {MODEL_PATH}"
)

print("\nTraining completed!")