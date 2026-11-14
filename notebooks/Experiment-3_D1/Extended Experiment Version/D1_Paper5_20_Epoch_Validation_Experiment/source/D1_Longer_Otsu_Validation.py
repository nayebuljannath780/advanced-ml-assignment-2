import csv
import hashlib
import io
import json
import os
import platform
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import tensorflow as tf
from IPython.display import FileLink, display
from PIL import Image, ImageOps, __version__ as pillow_version
from tensorflow import keras

size = 224
image_size = 224
d1_size = 224
d1_data_seed = 2026
d1_learning_rate = 1e-4
d1_momentum = 0.9
batch_size = 30
shared_hash = "7531bf3507abba811ae94207e0b4075124567ba239e09c772b402e5635c6dc50"


def mounted_dataset(slug):
    for parent in (Path("/kaggle/input/datasets/nayebul0"), Path("/kaggle/input")):
        candidate = parent / slug
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Attach this Kaggle input: nayebul0/{slug}")


def resize_to_fit(rgb):
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgb, dimensions, interpolation=interpolation)


def stretch_luminance(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    low = float(lightness.min())
    high = float(lightness.max())
    if high == low:
        return rgb.copy()
    lab[:, :, 0] = (lightness - low) * (100.0 / (high - low))
    stretched = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.rint(np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)


def bright_otsu_mask(gray):
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.int64)
    count_below = np.cumsum(histogram)[:-1]
    sum_below = np.cumsum(histogram * np.arange(256, dtype=np.int64))[:-1]
    count_above = gray.size - count_below
    valid = (count_below > 0) & (count_above > 0)
    if not valid.any():
        return np.ones(gray.shape, dtype=bool), None

    total_intensity = int(histogram @ np.arange(256, dtype=np.int64))
    separation = gray.size * sum_below[valid] - total_intensity * count_below[valid]
    scores = separation.astype(np.float64) ** 2 / (count_below[valid] * count_above[valid])
    threshold = int(np.flatnonzero(valid)[np.argmax(scores)] + 1)
    return gray >= threshold, threshold


def build_vgg_path(inputs):
    x = inputs
    for block_id, widths in enumerate(((64,), (128,), (256, 256), (512, 512)), start=1):
        for conv_id, width in enumerate(widths, start=1):
            x = layers.Conv2D(
                width, 3, padding="same", activation="relu",
                name=f"block{block_id}_conv{conv_id}"
            )(x)
        x = layers.MaxPooling2D(2, strides=2, name=f"block{block_id}_pool")(x)
    x = layers.Flatten(name="vgg_flatten")(x)
    return layers.Dense(
        512, activation="relu", kernel_initializer="he_normal", name="vgg_projection"
    )(x)


def residual_bottleneck(inputs, width, name, projection):
    shortcut = inputs
    if projection:
        shortcut = layers.Conv2D(4 * width, 1, name=f"{name}_0_conv")(shortcut)
        shortcut = layers.BatchNormalization(epsilon=1.001e-5, name=f"{name}_0_bn")(shortcut)

    x = inputs
    for conv_id, (filters, kernel_size) in enumerate(((width, 1), (width, 3), (4 * width, 1)), start=1):
        x = layers.Conv2D(
            filters, kernel_size, strides=1, padding="same", name=f"{name}_{conv_id}_conv"
        )(x)
        x = layers.BatchNormalization(epsilon=1.001e-5, name=f"{name}_{conv_id}_bn")(x)
        if conv_id < 3:
            x = layers.Activation("relu", name=f"{name}_{conv_id}_relu")(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    return layers.Activation("relu", name=f"{name}_out")(x)


def build_resnet_path(inputs):
    x = layers.ZeroPadding2D(3, name="conv1_pad")(inputs)
    x = layers.Conv2D(64, 7, strides=2, name="conv1_conv")(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.ZeroPadding2D(1, name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    blocks = (
        ("conv2_block1", 64, True, True),
        ("conv2_block2", 64, False, False),
        ("conv3_block1", 128, True, True),
        ("conv3_block2", 128, False, False),
        ("conv4_block1", 256, True, False)
    )
    for name, width, projection, downsample in blocks:
        x = residual_bottleneck(x, width, name, projection)
        if downsample:
            x = layers.MaxPooling2D(3, strides=2, padding="same", name=f"{name}_pool")(x)

    return layers.GlobalAveragePooling2D(name="resnet_gap")(x)


def self_gate(features, name):
    weights = layers.Activation("sigmoid", name=f"{name}_sigmoid")(features)
    return layers.Multiply(name=f"{name}_gated")([features, weights])


def build_amff(class_count):
    inputs = keras.Input(shape=(image_size, image_size, 3), dtype="float32", name="prepared_image")
    vgg = build_vgg_path(inputs)
    resnet = build_resnet_path(inputs)
    fusion = layers.Concatenate(name="amff_fusion")([
        self_gate(vgg, "vgg"),
        self_gate(resnet, "resnet")
    ])
    x = layers.Dense(
        512, activation="relu", kernel_initializer="he_normal", name="fusion_dense"
    )(fusion)
    x = layers.Dropout(0.5, name="fusion_dropout")(x)
    outputs = layers.Dense(class_count, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="amff_hybrid_d1")


def read_d1_manifest(manifest_path):
    raw = manifest_path.read_bytes()
    expected = "3b756d0a4f0326a5e5123284aa884f23fe604d4ace6c56477491acd80ae728d0"
    assert hashlib.sha256(raw).hexdigest() == expected, "D1 frozen manifest hash mismatch"
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def center_pad_d1(rgb):
    height, width = rgb.shape[:2]
    top = (d1_size - height) // 2
    left = (d1_size - width) // 2
    canvas = np.zeros((d1_size, d1_size, 3), dtype=np.uint8)
    canvas[top:top + height, left:left + width] = rgb
    return canvas


def prepare_d1_image(path):
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    stretched = stretch_luminance(resize_to_fit(rgb))
    gray = cv2.cvtColor(stretched, cv2.COLOR_RGB2GRAY)
    mask, _ = bright_otsu_mask(gray)
    masked = np.where(mask[:, :, None], stretched, np.uint8(0))
    return center_pad_d1(masked), center_pad_d1(stretched)


def d1_affine_matrix(epoch, row_index):
    rng = np.random.default_rng([d1_data_seed, epoch, int(row_index)])
    offset = rng.uniform(-25.0, 25.0, size=2)
    scales = rng.uniform(0.9, 1.1, size=2)
    angle = float(rng.uniform(-20.0, 20.0))
    center = ((d1_size - 1) / 2,) * 2
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    linear = rotation[:, :2] @ np.diag(scales)
    translation = np.asarray(center) + offset - linear @ center
    return np.column_stack((linear, translation))


def d1_imagenet_input(rgb_batch):
    batch = rgb_batch.astype(np.float32)
    return np.ascontiguousarray(keras.applications.vgg16.preprocess_input(batch))


class D1FitBatches(keras.utils.PyDataset):
    def __init__(self, images, labels, row_indices, training, batch_size):
        super().__init__(workers=1, use_multiprocessing=False)
        self.images = images
        self.labels = labels
        self.row_indices = row_indices
        self.training = training
        self.batch_size = batch_size
        self.epoch = 0
        self.order = self.epoch_order()

    def epoch_order(self):
        if self.training:
            return np.random.default_rng([d1_data_seed, self.epoch]).permutation(self.row_indices)
        return self.row_indices

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def batch_rows(self, index):
        start = index * self.batch_size
        return self.order[start:start + self.batch_size]

    def __getitem__(self, index):
        indices = self.batch_rows(index)
        rgb_batch = self.images[indices]
        if self.training:
            for position, row_index in enumerate(indices):
                rgb_batch[position] = cv2.warpAffine(
                    rgb_batch[position], d1_affine_matrix(self.epoch, row_index),
                    (d1_size, d1_size), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
                )
        return d1_imagenet_input(rgb_batch), self.labels[indices]

    def set_epoch(self, epoch):
        if self.training:
            self.epoch = epoch
            self.order = self.epoch_order()


class D1EpochOrder(keras.callbacks.Callback):
    def __init__(self, train_batches):
        super().__init__()
        self.train_batches = train_batches

    def on_epoch_begin(self, epoch, logs=None):
        self.train_batches.set_epoch(epoch)


def make_d1_optimizer():
    return keras.optimizers.SGD(
        learning_rate=d1_learning_rate, momentum=d1_momentum, nesterov=False
    )


def d1_classification_metrics(truth, probabilities):
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    correct = np.diag(confusion)
    assert support.min() > 0
    precision = np.divide(correct, predicted_count, out=np.zeros(8), where=predicted_count > 0)
    recall = correct / support
    f1 = 2 * correct / (support + predicted_count)
    return {
        "correct": int(correct.sum()),
        "accuracy": float(correct.sum() / len(truth)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "class_id": i, "support": int(support[i]), "correct": int(correct[i]),
                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])
            }
            for i in range(8)
        ]
    }


def save_d1_validation_predictions(path, rows, probabilities):
    fields = ["rel_path", "true_class_id", "predicted_class_id"] + [f"p_{i}" for i in range(8)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row, probability in zip(rows, probabilities, strict=True):
            writer.writerow([
                row["rel_path"], row["class_id"], int(probability.argmax()),
                *[f"{value:.9g}" for value in probability]
            ])


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_shared_initialization():
    name = "D1_amff_shared_init.weights.h5"
    live_file = Path("/kaggle/working/paper5_d1_initialization") / name
    candidates = [live_file, *sorted(Path("/kaggle/input").rglob(name))]
    for path in candidates:
        if path.is_file() and file_sha256(path) == shared_hash:
            return path
    raise FileNotFoundError(
        f"Attach the saved {name} as a Kaggle input. Required SHA-256: {shared_hash}"
    )


def run_d1_longer_otsu():
    initialization = find_shared_initialization()
    manifest = mounted_dataset("assignment02-verified-train-val-test-splits") / "D1_split_manifest.csv"
    image_root = mounted_dataset("assignment02-d1-original-rice-leaf-1701") / "Original Images"
    rows = [row for row in read_d1_manifest(manifest) if row["split"] in ("train", "val")]
    split_counts = Counter(row["split"] for row in rows)
    assert split_counts == {"train": 1156, "val": 247}, split_counts
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int32)
    split_indices = {
        split: np.asarray([i for i, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
        for split in ("train", "val")
    }
    class_names = {int(row["class_id"]): row["class_name"] for row in rows}
    print(f"Verified shared initialization: {shared_hash}")
    print("Preparing 1,156 training and 247 validation Otsu images.")
    images = np.empty((len(rows), 224, 224, 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        images[i] = prepare_d1_image(image_root / row["rel_path"])[0]
    training = D1FitBatches(images, labels, split_indices["train"], True, batch_size)
    validation = D1FitBatches(images, labels, split_indices["val"], False, batch_size)
    assert (len(training), len(validation)) == (39, 9)

    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("float32")
    keras.utils.set_random_seed(d1_data_seed)
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "Enable a GPU in this Kaggle notebook."
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    root = Path("/kaggle/working/paper5_d1_longer_validation")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="otsu_", dir=root))
    checkpoint = run_dir / "D1_longer_otsu_best_validation.weights.h5"
    epoch_log = run_dir / "epoch_log.csv"
    config_path = run_dir / "configuration.json"
    config_path.write_text(json.dumps({
        "dataset": "D1", "mode": "otsu", "evaluation_scope": "validation_only",
        "manifest_sha256": file_sha256(manifest), "initialization_sha256": shared_hash,
        "split_counts": dict(split_counts), "class_map": class_names, "input_shape": [224, 224, 3],
        "seed": d1_data_seed, "batch_size": batch_size, "max_epochs": 20,
        "early_stopping_patience": 4, "monitor": "val_loss", "optimizer": "SGD",
        "learning_rate": d1_learning_rate, "momentum": d1_momentum, "nesterov": False,
        "loss": "sparse categorical cross-entropy", "dtype": "float32", "jit_compile": False,
        "op_determinism": True, "gpu": gpu_details,
        "python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
        "numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print(f"GPU: {gpu_details} | TensorFlow {tf.__version__} | Keras {keras.__version__}")

    with tf.device("/GPU:0"):
        model = build_amff(8)
        model.load_weights(str(initialization))
        model.compile(
            optimizer=make_d1_optimizer(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            jit_compile=False
        )
        saver = keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min",
            save_weights_only=True, save_best_only=True, verbose=0
        )
        callbacks = [
            D1EpochOrder(training), keras.callbacks.TerminateOnNaN(), saver,
            keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0, patience=4),
            keras.callbacks.CSVLogger(str(epoch_log))
        ]
        history = model.fit(
            training, validation_data=validation, epochs=20, validation_freq=1,
            shuffle=False, verbose=2, callbacks=callbacks
        ).history
        losses = history["val_loss"]
        assert 1 <= len(losses) <= 20 and all(np.isfinite(values).all() for values in history.values())
        updates = int(model.optimizer.iterations.numpy())
        assert updates == len(losses) * len(training), (updates, len(losses))
        best = int(np.argmin(losses))
        np.testing.assert_allclose(saver.best, losses[best])
        model.load_weights(str(checkpoint))
        selected = model.evaluate(validation, verbose=0, return_dict=True)
        np.testing.assert_allclose(
            [selected["loss"], selected["accuracy"]],
            [losses[best], history["val_accuracy"][best]],
            atol=1e-6, rtol=1e-5
        )
        predicted = []
        observed = []
        for i in range(len(validation)):
            batch, truth = validation[i]
            predicted.append(model(batch, training=False).numpy())
            observed.append(truth)

    probabilities = np.concatenate(predicted)
    truth = np.concatenate(observed)
    np.testing.assert_array_equal(truth, labels[split_indices["val"]])
    assert probabilities.shape == (247, 8) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5)
    metrics = d1_classification_metrics(truth, probabilities)
    np.testing.assert_allclose(metrics["accuracy"], selected["accuracy"], rtol=0, atol=1e-7)

    baseline_losses = np.array([2.00555, 1.83013, 1.68451, 1.65185, 1.51528, 1.430259])
    baseline_correct = np.array([58, 89, 104, 109, 107, 121])
    first_six_match = bool(
        len(losses) >= 6
        and np.allclose(losses[:6], baseline_losses, rtol=0, atol=1e-5)
        and np.array_equal(np.rint(np.asarray(history["val_accuracy"][:6]) * 247).astype(int), baseline_correct)
    )
    baseline_f1 = 0.44414915118825776
    print(f"First six epochs match the original validation log: {first_six_match}")
    print(f"Best validation checkpoint: epoch {best + 1} | loss {losses[best]:.6f}")
    print(f"Validation accuracy: {metrics['accuracy']:.4%} ({metrics['correct']}/247)")
    print(f"Validation macro F1: {metrics['macro_f1']:.6f} | change vs six epochs: {(metrics['macro_f1'] - baseline_f1) * 100:+.2f} pp")
    for entry in metrics["per_class"]:
        entry["class_name"] = class_names[entry["class_id"]]
        print(f"{entry['class_name']} | recall {entry['recall']:.2%} | {entry['correct']}/{entry['support']}")

    prediction_path = run_dir / "validation_predictions.csv"
    save_d1_validation_predictions(prediction_path, [rows[i] for i in split_indices["val"]], probabilities)
    summary_path = run_dir / "validation_summary.json"
    summary_path.write_text(json.dumps({
        "epochs_completed": len(losses), "optimizer_updates": updates, "best_epoch": best + 1,
        "best_validation_loss": float(losses[best]), "first_six_match_original_log": first_six_match,
        "checkpoint_sha256": file_sha256(checkpoint), "original_six_epoch_macro_f1": baseline_f1,
        "validation_metrics": metrics
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    archive_path = run_dir / "D1_Longer_Otsu_Validation_Evidence.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in (config_path, epoch_log, prediction_path, summary_path):
            archive.write(path, arcname=path.name)
    with ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == 4 and archive.testzip() is None
    print(f"D1 LONGER OTSU VALIDATION PASSED | {len(losses)} epochs | {updates} updates")
    display(FileLink(os.path.relpath(archive_path)))
    display(FileLink(os.path.relpath(checkpoint)))


run_d1_longer_otsu()
