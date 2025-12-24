from ultralytics import YOLO
from pathlib import Path
import torch

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "yolo_datasets"
    / "Mergui.v4-version_for_tf.yolov8"
    / "data.yaml"
)

assert DATASET_YAML.exists(), f"Dataset YAML not found: {DATASET_YAML}"

# ============================================================
# Hardware check
# ============================================================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ============================================================
# Training configuration
# ============================================================
BASE_MODEL = "yolov8s.pt"   # better accuracy, still Hailo-friendly

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16 if DEVICE == "cpu" else 32
WORKERS = 8

OUTPUT_DIR = PROJECT_ROOT / "results" / "yolo_train_horse"

# ============================================================
# Train
# ============================================================
def train():
    print("[INFO] Loading COCO-pretrained model:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print("[INFO] Starting transfer learning on horse dataset...")
    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        optimizer="SGD",
        project=str(OUTPUT_DIR),
        name="yolov8s_horse_coco_tl",
        exist_ok=True,
        verbose=True,
        patience=20,          # early stopping
        cos_lr=True,          # smoother convergence
        mosaic=1.0,           # data augmentation
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
    )

    print("\n[OK] Training completed.")
    print("[INFO] Best model:")
    print(OUTPUT_DIR / "yolov8s_horse_coco_tl" / "weights" / "best.pt")


if __name__ == "__main__":
    train()
