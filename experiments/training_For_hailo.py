from ultralytics import YOLO
from pathlib import Path
import torch

# ============================================================
# PROJECT PATHS
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
# HARDWARE CHECK
# ============================================================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ============================================================
# TRAINING CONFIGURATION (HAILO-8L SAFE)
# ============================================================
BASE_MODEL = "yolov8n.pt"     # ✅ REQUIRED for Hailo-8L
IMG_SIZE = 512                # ✅ Safe resolution for Hailo-8L
EPOCHS = 150                  # a bit longer for nano model
BATCH_SIZE = 16 if DEVICE == "cpu" else 32
WORKERS = 8

OUTPUT_DIR = PROJECT_ROOT / "results" / "yolo_hailo8l_safe"

RUN_NAME = "yolov8n_horse_hailo8l_safe_512"

# ============================================================
# TRAIN
# ============================================================
def train():
    print("\n============================================================")
    print("[INFO] HAILO-8L SAFE TRAINING CONFIGURATION")
    print(f"       Model: {BASE_MODEL}")
    print(f"       Image size: {IMG_SIZE}")
    print("============================================================\n")

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
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        patience=30,           # early stopping
        cos_lr=True,           # smoother convergence

        # --------------------
        # Data augmentation
        # --------------------
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
    )

    best_model_path = (
        OUTPUT_DIR
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    print("\n[OK] Training completed successfully.")
    print("[INFO] Best model saved at:")
    print(best_model_path)


if __name__ == "__main__":
    train()
