"""
Butterfly Detector — YOLO11 ONNX inference over Windows Shared Memory.

Reads frames written by butterfly_capture.exe via IPC shared memory,
runs ONNX Runtime inference, and prints detections to stdout.
Each detection line is CSV-compatible for piping to TTS or overlay tools.

Usage:
    detector.exe [--model PATH] [--conf FLOAT] [--iou FLOAT] [--verbose]
"""

import sys
import os
import time
import math
import random
import threading
import ctypes
import argparse
import numpy as np

# Resolve ipc_config.py path:
# - When frozen by PyInstaller (onefile), modules live in sys._MEIPASS
# - When run as script, shared/ is two directories up
if getattr(sys, "frozen", False):
    _IPC_DIR = sys._MEIPASS
else:
    _IPC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared")
sys.path.insert(0, _IPC_DIR)

from ipc_config import (
    BUTTERFLY_FRAME_NAME, BUTTERFLY_READY_NAME,
    ButterflyFrameHeader, HEADER_SIZE, PIXEL_OFFSET,
    SHM_TOTAL_SIZE, BUTTERFLY_FRAME_MAGIC,
    BUTTERFLY_MAX_WIDTH, BUTTERFLY_MAX_HEIGHT
)

try:
    import onnxruntime as ort
    import cv2
except ImportError as e:
    print(f"[detector] Missing dependency: {e}", file=sys.stderr)
    print("[detector] Run: pip install onnxruntime opencv-python numpy", file=sys.stderr)
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Naked", "Cloth", "Leather", "RoadSign",
    "Metal", "Hazmat", "Heavy", "Scientist", "Head"
]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_INPUT_SIZE = 640

# ── Windows IPC helpers ────────────────────────────────────────────────────────

kernel32 = ctypes.windll.kernel32

# On 64-bit Windows all HANDLE / LPVOID returns are 64-bit pointers.
# Without explicit restype ctypes defaults to c_int (32-bit) and truncates
# the upper 32 bits — wrong address, immediate crash.
kernel32.OpenFileMappingW.restype  = ctypes.c_void_p
kernel32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p]

kernel32.MapViewOfFile.restype  = ctypes.c_void_p
kernel32.MapViewOfFile.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t,
]

kernel32.OpenSemaphoreW.restype  = ctypes.c_void_p
kernel32.OpenSemaphoreW.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p]

kernel32.WaitForSingleObject.restype  = ctypes.c_uint32
kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

kernel32.CloseHandle.restype  = ctypes.c_bool
kernel32.CloseHandle.argtypes = [ctypes.c_void_p]


def _open_file_mapping(name: str) -> int:
    FILE_MAP_READ = 0x0004
    h = kernel32.OpenFileMappingW(FILE_MAP_READ, False, name)
    if not h:
        raise OSError(f"OpenFileMappingW({name!r}) failed: {ctypes.get_last_error()}")
    return h


def _map_view(handle: int, size: int) -> int:
    FILE_MAP_READ = 0x0004
    ptr = kernel32.MapViewOfFile(handle, FILE_MAP_READ, 0, 0, size)
    if not ptr:
        raise OSError(f"MapViewOfFile failed: {ctypes.get_last_error()}")
    return ptr


def _open_semaphore(name: str) -> int:
    SEMAPHORE_MODIFY_STATE = 0x0002
    SYNCHRONIZE = 0x00100000
    h = kernel32.OpenSemaphoreW(SEMAPHORE_MODIFY_STATE | SYNCHRONIZE, False, name)
    if not h:
        raise OSError(f"OpenSemaphoreW({name!r}) failed: {ctypes.get_last_error()}")
    return h


def _wait_semaphore(handle: int, timeout_ms: int) -> bool:
    WAIT_OBJECT_0 = 0x00000000
    WAIT_TIMEOUT  = 0x00000102
    result = kernel32.WaitForSingleObject(handle, timeout_ms)
    if result == WAIT_OBJECT_0:
        return True
    if result == WAIT_TIMEOUT:
        return False
    raise OSError(f"WaitForSingleObject error: {ctypes.get_last_error()}")

# ── YOLO11 post-processing ────────────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int) -> tuple[np.ndarray, float, int, int]:
    """Resize image with letterboxing to fit inside target×target."""
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target - new_h
    pad_w = target - new_w
    top  = pad_h // 2
    left = pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, top, left


def decode_yolo_output(
        raw: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        scale: float,
        pad_top: int,
        pad_left: int,
        orig_w: int,
        orig_h: int,
) -> list[dict]:
    """
    Decode YOLO11 output tensor [1, 84, 8400] into list of detection dicts.
    Each dict: {class_id, class_name, confidence, x1, y1, x2, y2}
    """
    preds = raw[0]  # [84, 8400]

    boxes_cx     = preds[0]
    boxes_cy     = preds[1]
    boxes_w      = preds[2]
    boxes_h      = preds[3]
    class_scores = preds[4:]

    num_cls = min(NUM_CLASSES, class_scores.shape[0])
    class_ids  = np.argmax(class_scores[:num_cls], axis=0)
    class_conf = class_scores[:num_cls][class_ids, np.arange(class_scores.shape[1])]

    mask = class_conf >= conf_threshold
    if not mask.any():
        return []

    cx   = boxes_cx[mask]
    cy   = boxes_cy[mask]
    bw   = boxes_w[mask]
    bh   = boxes_h[mask]
    cids = class_ids[mask]
    conf = class_conf[mask]

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h)

    detections = []
    for cls in np.unique(cids):
        cls_mask  = cids == cls
        cls_boxes = np.stack([x1[cls_mask], y1[cls_mask],
                              x2[cls_mask] - x1[cls_mask],
                              y2[cls_mask] - y1[cls_mask]], axis=1)
        cls_confs = conf[cls_mask].tolist()
        indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(), cls_confs,
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )
        if len(indices) == 0:
            continue
        for idx in np.array(indices).flatten():
            detections.append({
                "class_id":   int(cls),
                "class_name": CLASS_NAMES[int(cls)],
                "confidence": float(cls_confs[idx]),
                "x1": int(x1[cls_mask][idx]),
                "y1": int(y1[cls_mask][idx]),
                "x2": int(x2[cls_mask][idx]),
                "y2": int(y2[cls_mask][idx]),
            })

    return sorted(detections, key=lambda d: d["confidence"], reverse=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Butterfly Vision — YOLO11 object detection on live screen capture"
    )
    parser.add_argument("--model", default="runs/rust_v1/weights/best.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default 0.45)")
    parser.add_argument("--show", action="store_true",
                        help="Open a window with boxes drawn over the captured frame")
    parser.add_argument("--verbose", action="store_true",
                        help="Print timing info to stderr")
    return parser.parse_args()


def build_session(model_path: str) -> ort.InferenceSession:
    if not os.path.exists(model_path):
        if getattr(sys, "frozen", False):
            bundled_path = os.path.join(sys._MEIPASS, "best.onnx")
            if os.path.exists(bundled_path):
                model_path = bundled_path

    if not os.path.exists(model_path):
        print(f"[detector] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # CUDA when the machine has it, CPU otherwise. The same model file runs
    # on both, so there is no second build to maintain.
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[detector] Using CUDA execution provider", file=sys.stderr)
    else:
        chosen = ["CPUExecutionProvider"]
        print("[detector] Using CPU execution provider", file=sys.stderr)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=chosen)


def draw_overlay(frame_rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Boxes and labels over a copy of the frame, for the preview window."""
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        cv2.rectangle(canvas, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 220, 120), 2)
        label = f"{d['class_name']} {d['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Подложка под подпись: на светлом кадре зелёный текст не читается.
        cv2.rectangle(canvas, (d["x1"], d["y1"] - th - 6),
                      (d["x1"] + tw + 6, d["y1"]), (0, 220, 120), -1)
        cv2.putText(canvas, label, (d["x1"] + 3, d["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()

    session = build_session(args.model)
    input_name = session.get_inputs()[0].name
    print(f"[detector] Model loaded: {args.model}", file=sys.stderr)
    print(f"[detector] conf={args.conf}  iou={args.iou}", file=sys.stderr)

    # Frames arrive from the capture process through shared memory: it writes,
    # we read, nobody copies the image through a socket or a file.
    print("[detector] Waiting for the capture process ...", file=sys.stderr)
    h_frame_map = h_semaphore = None
    while h_frame_map is None:
        try:
            h_frame_map = _open_file_mapping(BUTTERFLY_FRAME_NAME)
            h_semaphore = _open_semaphore(BUTTERFLY_READY_NAME)
        except OSError:
            time.sleep(0.5)

    frame_ptr = _map_view(h_frame_map, SHM_TOTAL_SIZE)
    print("[detector] Connected to shared memory", file=sys.stderr)

    frame_count = 0
    total_inf_ms = 0.0

    while True:
        # The semaphore is what makes this cheap: we sleep until a frame is
        # ready instead of polling the buffer.
        if not _wait_semaphore(h_semaphore, 5):
            continue

        hdr = ButterflyFrameHeader.from_address(frame_ptr)
        if hdr.magic != BUTTERFLY_FRAME_MAGIC:
            continue

        w = hdr.width
        h = hdr.height
        if w == 0 or h == 0:
            continue

        pixel_addr = frame_ptr + PIXEL_OFFSET
        pixel_bytes = w * h * 4
        buf = (ctypes.c_uint8 * pixel_bytes).from_address(pixel_addr)
        frame_bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        frame_rgb = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGB)

        lb_img, scale, pad_top, pad_left = letterbox(frame_rgb, MODEL_INPUT_SIZE)
        tensor = lb_img.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        inf_ms = (time.perf_counter() - t0) * 1000.0

        detections = decode_yolo_output(
            outputs[0], args.conf, args.iou,
            scale, pad_top, pad_left, w, h
        )

        frame_count += 1
        total_inf_ms += inf_ms

        if detections:
            # One line per frame plus one per box: the format is CSV-friendly,
            # so another program can read it from stdout.
            print(f"[detector] frame={hdr.frame_id} dets={len(detections)} inf={inf_ms:.1f}ms")
            for d in detections:
                print(f"  {d['class_name']:<12} conf={d['confidence'] * 100:.0f}%"
                      f"  box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")
            sys.stdout.flush()
        elif args.verbose:
            print(f"[detector] frame={hdr.frame_id}  no detections  inf={inf_ms:.1f}ms",
                  file=sys.stderr)

        if args.show:
            cv2.imshow("butterfly-vision", draw_overlay(frame_rgb, detections))
            if cv2.waitKey(1) & 0xFF == 27:   # Esc
                break

        if args.verbose and frame_count % 100 == 0:
            print(f"[detector] avg inference={total_inf_ms / frame_count:.1f}ms "
                  f"over {frame_count} frames", file=sys.stderr)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
