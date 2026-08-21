from __future__ import annotations

import argparse
import contextlib
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MODEL_SIZE = 640
DFL_BINS   = 16        
CONF       = 0.5
IOU        = 0.45
CLASSES    = ["aircraft"]
_EPS       = 1e-6


def _radial_M(r: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    r = np.clip(r, 0.0, 1.0 - _EPS)
    beta = min(max(beta, 0.05), 0.98)
    alpha = min(max(alpha, 0.0), 1.0)
    poincare = np.arctanh(beta * r) / np.arctanh(beta)
    return (1.0 - alpha) * r + alpha * poincare


def unwarp_boxes(boxes_xyxy: np.ndarray, alpha: float, beta: float,size: int, n_edge: int = 8) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    out = np.empty_like(boxes_xyxy, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        xs = np.concatenate([x1 + t * (x2 - x1), x1 + t * (x2 - x1),
                             np.full(n_edge, x1), np.full(n_edge, x2)])
        ys = np.concatenate([np.full(n_edge, y1), np.full(n_edge, y2),
                             y1 + t * (y2 - y1), y1 + t * (y2 - y1)])
        xn = xs / (size - 1) * 2.0 - 1.0          # pixel -> [-1,1]
        yn = ys / (size - 1) * 2.0 - 1.0
        r = np.sqrt(xn * xn + yn * yn)
        r_safe = np.clip(r, _EPS, None)
        r_new = _radial_M(np.clip(r, 0.0, 1.0), alpha, beta)
        scale = np.where(r <= 1.0, r_new / r_safe, 1.0)
        xp = (xn * scale + 1.0) * 0.5 * (size - 1)
        yp = (yn * scale + 1.0) * 0.5 * (size - 1)
        out[i] = [xp.min(), yp.min(), xp.max(), yp.max()]
    return out


def letterbox(img: np.ndarray, size: int = MODEL_SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top

def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_v8(outputs, conf_thres):
    feats = {}
    for arr in outputs.values():
        a = np.squeeze(arr)
        if a.ndim == 2:
            a = a[..., None]
        h, _, c = a.shape
        feats.setdefault(h, {})[("box" if c == DFL_BINS * 4 else "cls")] = a

    bins = np.arange(DFL_BINS, dtype=np.float32)
    boxes_all, scores_all = [], []
    for size, fc in sorted(feats.items()):
        if "box" not in fc or "cls" not in fc:
            raise RuntimeError(f"feature size {size} missing box/cls output")
        stride = MODEL_SIZE // size

        box = fc["box"].reshape(-1, 4, DFL_BINS).astype(np.float32)
        dist = (_softmax(box, axis=-1) * bins).sum(-1)        # (N,4) l,t,r,b

        gy, gx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        ax = gx.reshape(-1).astype(np.float32) + 0.5
        ay = gy.reshape(-1).astype(np.float32) + 0.5

        x1 = (ax - dist[:, 0]) * stride
        y1 = (ay - dist[:, 1]) * stride
        x2 = (ax + dist[:, 2]) * stride
        y2 = (ay + dist[:, 3]) * stride

        score = _sigmoid(fc["cls"].reshape(-1).astype(np.float32))
        keep = score >= conf_thres
        boxes_all.append(np.stack([x1, y1, x2, y2], axis=1)[keep])
        scores_all.append(score[keep])

    if not boxes_all or all(len(b) == 0 for b in boxes_all):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return (np.concatenate(boxes_all).astype(np.float32),
            np.concatenate(scores_all).astype(np.float32))


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


class HailoYOLOv8:
    def __init__(self, hef_path: str):
        from hailo_platform import (HEF, VDevice, ConfigureParams,HailoStreamInterface, InferVStreams,InputVStreamParams, OutputVStreamParams,FormatType)
        self._stack = contextlib.ExitStack()
        self.hef = HEF(hef_path)
        self.target = self._stack.enter_context(VDevice())
        cfg = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.target.configure(self.hef, cfg)[0]
        ng_params = self.ng.create_params()
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_infos = self.hef.get_output_vstream_infos()
        in_params = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32)
        self.pipe = self._stack.enter_context(
            InferVStreams(self.ng, in_params, out_params))
        self._stack.enter_context(self.ng.activate(ng_params))
        self.input_hw = tuple(self.in_info.shape[:2])

    def infer(self, rgb_uint8):
        return self.pipe.infer({self.in_info.name: rgb_uint8[np.newaxis, ...]})

    def close(self):
        self._stack.close()


class CSICamera:
    def __init__(self, width, height):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}))
        self.cam.start()
        time.sleep(0.5)

    def read(self):
        return True, self.cam.capture_array()

    def release(self):
        self.cam.stop()


class USBCamera:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open USB camera index {index}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def run_frame(model, frame, fov, conf, iou):
    map_x, map_y, alpha, beta, imgsz = fov
    lb, ratio, pad_l, pad_t = letterbox(frame, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    foveated = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

    t0 = time.perf_counter()
    outputs = model.infer(np.ascontiguousarray(foveated))
    infer_ms = (time.perf_counter() - t0) * 1000.0

    boxes, scores = decode_v8(outputs, conf)
    keep = nms(boxes, scores, iou)
    boxes, scores = boxes[keep], scores[keep]

    if len(boxes):
        boxes = unwarp_boxes(boxes, alpha, beta, imgsz)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / ratio
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    return boxes, scores, infer_ms, outputs


def draw(frame, boxes, scores, infer_ms):
    for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"aircraft {s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    infer_fps = 1000.0 / max(infer_ms, 1e-6)
    cv2.putText(frame, f"infer {infer_ms:4.1f}ms  ({infer_fps:5.1f} FPS)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def probe(outputs):
    print("--- HEF raw outputs ---")
    for name, arr in outputs.items():
        a = np.squeeze(arr)
        print(f"  {name:20s} shape={a.shape}  dtype={arr.dtype}  "
              f"min={a.min():.3f}  max={a.max():.3f}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hef", default="best.hef")
    p.add_argument("--fov", default="telescope_foveation.npz",
                   help="foveation LUT (map_x, map_y, alpha, beta, imgsz)")
    p.add_argument("--image", default=None,
                   help="run on a single image (or folder) instead of a camera")
    p.add_argument("--camera", choices=("csi", "usb"), default="usb")
    p.add_argument("--usb-index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--iou", type=float, default=IOU)
    p.add_argument("--probe", action="store_true",
                   help="print raw HEF output shapes/ranges and exit")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save", default=None, help="save annotated image(s) here")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out-dir", default="recordings",
                   help="directory where the mp4 recording is saved")
    p.add_argument("--fps", type=float, default=20.0,
                   help="fps written into the recording file")
    p.add_argument("--no-record", action="store_true",
                   help="do not save an mp4 recording of the session")
    return p.parse_args()


def load_fov(path):
    f = np.load(path)
    return (f["map_x"], f["map_y"], float(f["alpha"]),
            float(f["beta"]), int(f["imgsz"]))


def run_images(model, fov, args):
    src = Path(args.image)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = ([src] if src.is_file()
            else sorted(p for p in src.rglob("*") if p.suffix.lower() in exts))
    if not imgs:
        raise SystemExit(f"no images at {src}")
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"skip unreadable {p}")
            continue
        boxes, scores, infer_ms, outputs = run_frame(
            model, frame, fov, args.conf, args.iou)
        if args.probe:
            probe(outputs)
            return
        dets = " ".join(f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]{s:.2f}"
                         for (x1, y1, x2, y2), s in zip(boxes, scores))
        print(f"{p.name}: {len(boxes)} det  {infer_ms:.1f}ms  {dets}")
        if args.save:
            draw(frame, boxes, scores, infer_ms)
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / p.name), frame)


def run_camera(model, fov, args):
    cam = (CSICamera(args.width, args.height) if args.camera == "csi"
           else USBCamera(args.usb_index, args.width, args.height))
    print(f"camera: {args.camera}  -- Ctrl+C / 'q' to stop")

    # mp4 recording (open lazily on the first frame so we know W,H).
    writer, out_path = None, None
    if not args.no_record:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"telescope_hailo_{stamp}.mp4"

    n, infer_ms_smooth = 0, 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("camera read failed")
                break
            if args.max_frames and n >= args.max_frames:
                break

            if writer is None and out_path is not None:
                H, W = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (W, H))
                if not writer.isOpened():
                    print(f"could not open VideoWriter {out_path} - recording off")
                    writer, out_path = None, None
                else:
                    print(f"Recording to: {out_path}")

            boxes, scores, infer_ms, outputs = run_frame(
                model, frame, fov, args.conf, args.iou)
            if args.probe:
                probe(outputs)
                return
            # Smooth the inference time so the on-screen FPS is steady.
            infer_ms_smooth = (infer_ms if n == 0
                               else 0.9 * infer_ms_smooth + 0.1 * infer_ms)

            # Always draw before recording so the saved mp4 has the overlays --
            # whether or not the window is shown.
            draw(frame, boxes, scores, infer_ms_smooth)
            if writer is not None:
                writer.write(frame)

            if args.no_display:
                if len(boxes):
                    print(f"f{n:05d}  infer {infer_ms_smooth:5.1f}ms "
                          f"({1000.0 / max(infer_ms_smooth, 1e-6):5.1f} FPS)  "
                          f"{len(boxes)} det")
            else:
                cv2.imshow("telescope Hailo-8 detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cam.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"done -- {n} frames")
        if out_path is not None:
            print(f"saved: {out_path}")


def main():
    args = parse_args()
    if not Path(args.hef).exists():
        raise FileNotFoundError(f"HEF not found: {args.hef}")
    if not Path(args.fov).exists():
        raise FileNotFoundError(f"foveation LUT not found: {args.fov}")

    fov = load_fov(args.fov)
    print(f"foveation: alpha={fov[2]:.4f} beta={fov[3]:.4f} imgsz={fov[4]}")
    print(f"loading HEF: {args.hef}")
    model = HailoYOLOv8(args.hef)
    if model.input_hw != (fov[4], fov[4]):
        print(f"WARNING: HEF input {model.input_hw} != foveation imgsz {fov[4]}")
    try:
        if args.image:
            run_images(model, fov, args)
        else:
            run_camera(model, fov, args)
    finally:
        model.close()


if __name__ == "__main__":
    main()
