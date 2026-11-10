"""
Local YOLO11 training script.
For GPU training prefer train_colab.ipynb on Google Colab T4.

Usage:
    python train.py [--model yolo11n.pt] [--epochs 50] [--batch 16]
"""
import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolo11n.pt",
                        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
                        help="Base YOLO11 model (n=nano fastest, l=large most accurate)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="0",
                        help="Device: 0 for GPU, cpu for CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[train] Install: pip install ultralytics")
        raise

    data_yaml = str(Path(__file__).parent / "rust.yaml")
    if not os.path.exists(data_yaml):
        print(f"[train] Dataset config not found: {data_yaml}")
        raise FileNotFoundError(data_yaml)

    model = YOLO(args.model)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="rust_v1",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )

    # Export to ONNX for deployment
    best_pt = str(Path("runs/rust_v1/weights/best.pt"))
    if os.path.exists(best_pt):
        print(f"[train] Exporting {best_pt} → ONNX ...")
        best_model = YOLO(best_pt)
        best_model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True)
        print("[train] Done: runs/rust_v1/weights/best.onnx")
    else:
        print(f"[train] WARNING: {best_pt} not found, export skipped")


if __name__ == "__main__":
    main()
