"""
train_yolo.py

Fine-tunes the existing Connect Four YOLO model on your own labeled data.

Prerequisites:
    pip install ultralytics
    # Place your Roboflow YOLOv8 export at: training_data/connect4_v2/
    # (should contain data.yaml, images/, labels/ subdirectories)

Usage:
    python train_yolo.py

Output:
    runs/detect/fine_tune/weights/best.pt   — best checkpoint

After training:
    cp runs/detect/fine_tune/weights/best.pt yolo-detect/detect/train/weights/best_v2.pt
"""

import random
import shutil
import tempfile
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Missing dependency: pip install ultralytics")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = Path("yolo-detect/detect/train/weights/best.pt")
DATASET_YAML = Path("/Users/eithan/Downloads/c4/finetune-yolo26/data.yaml")
VAL_FRACTION = 0.1   # fraction of train images held out for validation
RANDOM_SEED  = 42
EPOCHS   = 50
IMG_SIZE = 640
BATCH    = 16        # reduce to 8 if OOM
DEVICE   = "mps"     # Apple Silicon GPU; set to "cpu" if issues arise
PROJECT  = "runs"
NAME     = "fine_tune"
# ─────────────────────────────────────────────────────────────────────────────


def _make_val_split(dataset_yaml: Path, val_fraction: float, tmp_dir: Path) -> Path:
    """
    Roboflow exported only a train/ split.  This function carves out a random
    val_fraction of the training images into a temporary valid/ directory and
    writes a corrected data.yaml pointing at both splits.

    Returns the path to the corrected data.yaml (inside tmp_dir).
    """
    train_images = dataset_yaml.parent / "train" / "images"
    train_labels = dataset_yaml.parent / "train" / "labels"

    all_images = sorted(train_images.glob("*.jpg")) + sorted(train_images.glob("*.png"))
    if not all_images:
        raise SystemExit(f"No images found in {train_images}")

    random.seed(RANDOM_SEED)
    n_val = max(1, int(len(all_images) * val_fraction))
    val_images = set(random.sample(all_images, n_val))
    print(f"Val split: {n_val}/{len(all_images)} images held out for validation")

    val_img_dir = tmp_dir / "valid" / "images"
    val_lbl_dir = tmp_dir / "valid" / "labels"
    val_img_dir.mkdir(parents=True)
    val_lbl_dir.mkdir(parents=True)

    for img_path in val_images:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        lbl_path = train_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, val_lbl_dir / lbl_path.name)

    # Write corrected data.yaml with absolute paths
    with open(dataset_yaml) as f:
        orig = yaml.safe_load(f)

    corrected = {
        "train": str(train_images.resolve()),
        "val":   str(val_img_dir.resolve()),
        "nc":    orig["nc"],
        "names": orig["names"],
    }
    corrected_yaml = tmp_dir / "data.yaml"
    with open(corrected_yaml, "w") as f:
        yaml.dump(corrected, f)

    return corrected_yaml


def main() -> None:
    if not BASE_MODEL.exists():
        raise SystemExit(f"Base model not found: {BASE_MODEL.resolve()}")
    if not DATASET_YAML.exists():
        raise SystemExit(f"Dataset yaml not found: {DATASET_YAML.resolve()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        effective_yaml = _make_val_split(DATASET_YAML, VAL_FRACTION, tmp_dir)

        print(f"Fine-tuning from: {BASE_MODEL}")
        print(f"Dataset: {DATASET_YAML}  (val split written to {tmp_dir})")
        print(f"Epochs: {EPOCHS}, imgsz: {IMG_SIZE}, batch: {BATCH}")
        print()

        model = YOLO(str(BASE_MODEL))
        results = model.train(
            data=str(effective_yaml),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            project=PROJECT,
            name=NAME,
            device=DEVICE,
            exist_ok=True,
            # Augmentations — add variation for robustness to lighting/camera changes
            hsv_h=0.02,
            hsv_s=0.5,
            hsv_v=0.4,
            fliplr=0.5,
            translate=0.1,
            scale=0.3,
            erasing=0.2,
        )

    # Ultralytics saves to <save_dir>/weights/best.pt; read actual path from results.
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    best_weights = (save_dir / "weights" / "best.pt") if save_dir else None

    print(f"\nTraining complete.")
    if best_weights and best_weights.exists():
        print(f"Best weights: {best_weights.resolve()}")
    else:
        print("Best weights location unknown — check the 'Logging results to' line above.")

    # Report final mAP
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    map50 = metrics.get("metrics/mAP50(B)", "N/A")
    map5095 = metrics.get("metrics/mAP50-95(B)", "N/A")
    print(f"Final mAP50: {map50}   mAP50-95: {map5095}")

    if best_weights:
        print(f"\nCopy to v2 slot:")
        print(f"  cp {best_weights} yolo-detect/detect/train/weights/best_v2.pt")


if __name__ == "__main__":
    main()
