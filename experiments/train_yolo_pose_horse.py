from ultralytics import YOLO
from pathlib import Path
import torch

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POSE_DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "yolo_datasets"
    / "keypoints"
    / "Horses.v5i.yolov8"
    / "data.yaml"
)

assert POSE_DATASET_YAML.exists(), f"Pose dataset YAML not found: {POSE_DATASET_YAML}"

# ============================================================
# Hardware
# ============================================================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ============================================================
# Training configuration (POSE)
# ============================================================
BASE_MODEL = "yolov8s-pose.pt"   # good accuracy for keypoints
EPOCHS = 120
IMG_SIZE = 640
BATCH_SIZE = 16 if DEVICE == "cpu" else 32
WORKERS = 8

OUTPUT_DIR = PROJECT_ROOT / "results" / "yolo_pose_horse"

# ============================================================
# Train
# ============================================================
def train_pose():
    print("[INFO] Loading COCO-pretrained pose model:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print("[INFO] Starting pose (keypoint) training...")
    model.train(
        data=str(POSE_DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        optimizer="SGD",
        project=str(OUTPUT_DIR),
        name="yolov8s_horse_pose",
        exist_ok=True,
        verbose=True,
        patience=25,          # early stopping
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
    )

    print("\n[OK] Pose training completed.")
    print("[INFO] Best pose model saved at:")
    print(OUTPUT_DIR / "yolov8s_horse_pose" / "weights" / "best.pt")


if __name__ == "__main__":
    train_pose()
