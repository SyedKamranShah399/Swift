from __future__ import annotations
import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,ConfigureParams, InputVStreamParams,OutputVStreamParams, FormatType)

HEF_PATH   = Path(r"yolov8n.hef")
CLASSES    = ["aircraft"]
MODEL_SIZE = 640
CONF       = 0.5


def letterbox(frame, size):
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    padx, pady = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[pady:pady + nh, padx:padx + nw] = resized
    return out, scale, padx, pady


def parse_nms(raw, scale, padx, pady, frame_w, frame_h, conf_thresh):
    dets = []
    per_class = raw[0] if isinstance(raw, (list, tuple)) else raw
    for cls_idx, cls_dets in enumerate(per_class):
        if cls_dets is None or len(cls_dets) == 0:
            continue
        for row in cls_dets:
            ymin, xmin, ymax, xmax, score = (float(v) for v in row[:5])
            if score < conf_thresh:
                continue
            x1 = (xmin * MODEL_SIZE - padx) / scale
            y1 = (ymin * MODEL_SIZE - pady) / scale
            x2 = (xmax * MODEL_SIZE - padx) / scale
            y2 = (ymax * MODEL_SIZE - pady) / scale
            x1 = int(max(0, min(frame_w - 1, x1)))
            y1 = int(max(0, min(frame_h - 1, y1)))
            x2 = int(max(0, min(frame_w,     x2)))
            y2 = int(max(0, min(frame_h,     y2)))
            if x2 > x1 and y2 > y1:
                dets.append((x1, y1, x2, y2, score, cls_idx))
    return dets


def open_camera(index, width, height):
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index} - try --camera 1")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hef",    default=str(HEF_PATH), help="path to the compiled .hef")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--conf",   type=float, default=CONF, help="score threshold")
    p.add_argument("--no-show", action="store_true", help="headless: print fps only")
    return p.parse_args()


def main():
    args = parse_args()
    hef_path = Path(args.hef)
    if not hef_path.exists():
        raise FileNotFoundError(
            f"HEF not found: {hef_path}\nCompile it first with compile_hef.sh.")

    print(f"HEF:    {hef_path}")
    hef = HEF(str(hef_path))
    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, cfg)[0]
        ng_params = network_group.create_params()

        in_info  = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        in_params  = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        print(f"input vstream : {in_info.name}  shape={in_info.shape}")
        print(f"output vstream: {out_info.name}")

        cap = open_camera(args.camera, args.width, args.height)
        print("Camera opened. Press 'q' to quit." if not args.no_show else "Camera opened (headless). Ctrl+C to quit.")

        prev, fps, infer_ms = time.time(), 0.0, 0.0
        with InferVStreams(network_group, in_params, out_params) as pipeline:
            with network_group.activate(ng_params):
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            print("\nno frame from camera")
                            break
                        H, W = frame.shape[:2]

                        lb, scale, padx, pady = letterbox(frame, MODEL_SIZE)
                        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
                        inp = np.expand_dims(rgb, axis=0)

                        t_inf = time.perf_counter()
                        results = pipeline.infer({in_info.name: inp})
                        infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t_inf) * 1000.0
                        dets = parse_nms(results[out_info.name],scale, padx, pady, W, H, args.conf)

                        for x1, y1, x2, y2, score, cls in dets:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{CLASSES[cls] if cls < len(CLASSES) else cls} {score:.2f}"
                            cv2.putText(frame, label, (x1, max(0, y1 - 6)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        now = time.time()
                        dt  = now - prev
                        prev = now
                        if dt > 0:
                            fps = 0.9 * fps + 0.1 * (1.0 / dt)

                        if args.no_show:
                            print(f"\rloop {fps:6.1f} FPS | infer {infer_ms:5.1f} ms "
                                  f"({1000.0 / max(infer_ms, 1e-6):6.1f} FPS) | "
                                  f"{len(dets)} det   ", end="", flush=True)
                        else:
                            cv2.putText(frame, f"loop {fps:5.1f} FPS  |  {len(dets)} det",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"infer {infer_ms:.1f} ms "
                                        f"({1000.0 / max(infer_ms, 1e-6):.0f} FPS)",
                                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.imshow("Hailo-8 detection - live", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                except KeyboardInterrupt:
                    pass
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nstopped.")


if __name__ == "__main__":
    main()
