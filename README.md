# Butterfly Vision — real-time object detection on screen capture

YOLO11 running on live screen frames: a C++ capture process grabs the screen
through Desktop Duplication, hands frames to Python over shared memory, and
ONNX Runtime detects objects in them — fast enough to keep up with the
screen.

```
capture.exe  ──shared memory──▶  detector.py  ──▶  boxes on stdout / overlay window
   C++              + semaphore        Python
```

## Why two processes and shared memory

Screen capture in Python is slow: every frame crosses the interpreter, gets
copied, and the GIL is held while it happens. So capture lives in a small C++
process that writes BGRA pixels straight into a shared memory block, and
signals a semaphore when a frame is ready.

The Python side maps the same block and reads it in place — `np.frombuffer`
over the mapped address, no copy. It waits on the semaphore instead of
polling, so an idle screen costs nothing.

That split is the interesting part of this project: the model is standard,
the frame path is not.

## The inference path

1. **Letterbox** the frame to the model input size, keeping aspect ratio, and
   remember scale and padding
2. **Normalise** to `float32 / 255` and transpose to `NCHW`
3. **Run** ONNX Runtime — CUDA when the machine has it, CPU otherwise, from
   the same model file
4. **Decode** raw YOLO output: confidence filter, then class-wise NMS
5. **Map boxes back** to screen coordinates, undoing the letterbox

Detections go to stdout in a format another program can read, and `--show`
opens a preview window with boxes drawn over the frame.

## Running

```
pip install -r detector/requirements.txt
cmake -S capture -B capture/build && cmake --build capture/build --config Release

capture.exe                                  # starts producing frames
python detector/detector.py --show           # consumes them
```

| Flag | Meaning |
|---|---|
| `--model` | Path to the `.onnx` file |
| `--conf` | Confidence threshold (default 0.25) |
| `--iou` | NMS IoU threshold (default 0.45) |
| `--show` | Preview window with boxes; Esc closes it |
| `--verbose` | Timing to stderr, average inference every 100 frames |

## Training

`detector/train.py` trains YOLO11 on a custom dataset described by
`detector/rust.yaml` (Ultralytics format). The trained weights are then
exported to ONNX, which is what the detector loads — one file that runs on
both CPU and CUDA without a second build.

The trained weights are not in this repository: a model file is large and
belongs in releases, not in git history.

## Scope

This repository is the computer-vision half only: capture, inference,
decoding and display. Anything that acts on the detections — moving an input
device, automating a game — is deliberately not here.

Windows. Python 3.11+, ONNX Runtime, OpenCV, NumPy; C++17 and CMake for the
capture process.
