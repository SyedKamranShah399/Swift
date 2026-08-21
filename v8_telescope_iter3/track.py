from __future__ import annotations

import argparse
import contextlib
import enum
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,ConfigureParams, InputVStreamParams,OutputVStreamParams, FormatType)

MODEL_SIZE = 640
DFL_BINS   = 16
CONF       = 0.50
IOU        = 0.45
_EPS       = 1e-6


def _radial_M(r, alpha, beta):
    r = np.clip(r, 0.0, 1.0 - _EPS)
    beta = min(max(beta, 0.05), 0.98)
    alpha = min(max(alpha, 0.0), 1.0)
    poincare = np.arctanh(beta * r) / np.arctanh(beta)
    return (1.0 - alpha) * r + alpha * poincare


def unwarp_boxes(boxes_xyxy, alpha, beta, size, n_edge=8):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    out = np.empty_like(boxes_xyxy, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        xs = np.concatenate([x1 + t * (x2 - x1), x1 + t * (x2 - x1),
                             np.full(n_edge, x1), np.full(n_edge, x2)])
        ys = np.concatenate([np.full(n_edge, y1), np.full(n_edge, y2),
                             y1 + t * (y2 - y1), y1 + t * (y2 - y1)])
        xn = xs / (size - 1) * 2.0 - 1.0
        yn = ys / (size - 1) * 2.0 - 1.0
        r = np.sqrt(xn * xn + yn * yn)
        r_safe = np.clip(r, _EPS, None)
        r_new = _radial_M(np.clip(r, 0.0, 1.0), alpha, beta)
        scale = np.where(r <= 1.0, r_new / r_safe, 1.0)
        xp = (xn * scale + 1.0) * 0.5 * (size - 1)
        yp = (yn * scale + 1.0) * 0.5 * (size - 1)
        out[i] = [xp.min(), yp.min(), xp.max(), yp.max()]
    return out


def letterbox(frame, size):
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[pady:pady + nh, padx:padx + nw] = resized
    return out, scale, padx, pady


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
        dist = (_softmax(box, axis=-1) * bins).sum(-1)
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
        self._stack = contextlib.ExitStack()
        self.hef = HEF(hef_path)
        self.target = self._stack.enter_context(VDevice())
        cfg = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.target.configure(self.hef, cfg)[0]
        ng_params = self.ng.create_params()
        self.in_info = self.hef.get_input_vstream_infos()[0]
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


def run_detector(model, frame, fov, conf, iou):
    H, W = frame.shape[:2]
    lb, scale, padx, pady = letterbox(frame, MODEL_SIZE)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    net_in = rgb
    if fov is not None:
        map_x, map_y, _, _, _ = fov
        net_in = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

    t0 = time.perf_counter()
    outputs = model.infer(np.ascontiguousarray(net_in))
    infer_ms = (time.perf_counter() - t0) * 1000.0

    boxes, scores = decode_v8(outputs, conf)
    keep = nms(boxes, scores, iou)
    boxes, scores = boxes[keep], scores[keep]
    if fov is not None and len(boxes):
        boxes = unwarp_boxes(boxes, fov[2], fov[3], fov[4])

    dets = []
    for (bx1, by1, bx2, by2), s in zip(boxes, scores):
        x1 = max(0.0, min(W - 1.0, (bx1 - padx) / scale))
        y1 = max(0.0, min(H - 1.0, (by1 - pady) / scale))
        x2 = max(0.0, min(float(W), (bx2 - padx) / scale))
        y2 = max(0.0, min(float(H), (by2 - pady) / scale))
        if x2 > x1 and y2 > y1:
            dets.append(((x1, y1, x2, y2), float(s)))
    return dets, infer_ms

@dataclass
class TrackOutput:
    state: str
    bbox: Optional[tuple]
    driver: str
    appearance: float
    match: float 

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter + 1e-9)


def _center(b): 
    return ((b[0] + b[2]) *0.5, (b[1]+b[3]))

def _center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


class LKState(enum.Enum):
    searching = 0
    locked = 1


_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class LKSingleTargetTracker:
    def __init__(self, estimate_scale: bool = False, template_size: int = 24,max_consecutive_misses: int = 10):
        self.estimate_scale = estimate_scale
        self.template_size = template_size
        self.max_consecutive_misses = max_consecutive_misses
        self._size: Optional[tuple] = None      # (w, h) reference size
        self._reset(keep_signature=False)

    def _reset(self, keep_signature: bool = True):
        self.state = LKState.searching
        self.bbox: Optional[tuple] = None
        self._pts: Optional[np.ndarray] = None
        self.misses = 0
        if not keep_signature:
            self._size = None

    def _seed_points(self, gray, bbox):
        x1, y1, x2, y2 = bbox
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i = min(gray.shape[1], int(x2))
        y2i = min(gray.shape[0], int(y2))
        if x2i - x1i < 6 or y2i - y1i < 6:
            return None
        roi = gray[y1i:y2i, x1i:x2i]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=80, qualityLevel=0.01,
                                      minDistance=4)
        if pts is None or len(pts) < 6:
            gx = np.linspace(x1i + 2, x2i - 2, 6)
            gy = np.linspace(y1i + 2, y2i - 2, 6)
            mx, my = np.meshgrid(gx, gy)
            grid = np.stack([mx.ravel(), my.ravel()], axis=1)
            return grid.astype(np.float32).reshape(-1, 1, 2)
        pts = pts.reshape(-1, 2)
        pts[:, 0] += x1i
        pts[:, 1] += y1i
        return pts.astype(np.float32).reshape(-1, 1, 2)

    def update_from_detection(self, gray, bbox) -> bool:
        pts = self._seed_points(gray, bbox)
        if pts is None:
            return False
        self.bbox = tuple(float(v) for v in bbox)
        self._pts = pts
        self._size = (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
        self.state = LKState.locked
        self.misses = 0
        return True

    def update(self, prev_gray, curr_gray):
        if self.state != LKState.locked or self._pts is None \
                or self.bbox is None:
            return
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, self._pts, None, **_LK_PARAMS)
        if nxt is None or st is None:
            self._register_miss()
            return
        ok = st.ravel() == 1
        good_old = self._pts[ok].reshape(-1, 2)
        good_new = nxt[ok].reshape(-1, 2)
        if len(good_new) < 4:
            self._register_miss()
            return

        disp = good_new - good_old
        dx = float(np.median(disp[:, 0]))
        dy = float(np.median(disp[:, 1]))

        scale = 1.0
        if self.estimate_scale and len(good_new) >= 6:
            c_old = good_old.mean(axis=0)
            c_new = good_new.mean(axis=0)
            r_old = np.linalg.norm(good_old - c_old, axis=1)
            r_new = np.linalg.norm(good_new - c_new, axis=1)
            m = r_old > 1e-3
            if np.count_nonzero(m) >= 4:
                scale = float(np.clip(np.median(r_new[m] / r_old[m]), 0.8, 1.25))

        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) * 0.5 + dx
        cy = (y1 + y2) * 0.5 + dy
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        self.bbox = (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)
        self._pts = good_new.astype(np.float32).reshape(-1, 1, 2)
        self.misses = 0

    def _register_miss(self):
        self.misses += 1
        if self.misses > self.max_consecutive_misses:
            self._reset(keep_signature=True)

    def size_consistent(self, bbox) -> bool:
        """True if bbox is within a sane scale band of the tracked size."""
        if self._size is None:
            return True
        rw, rh = self._size
        if rw <= 1.0 or rh <= 1.0:
            return True
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        ratio = np.sqrt((w * h) / (rw * rh))
        return 0.5 <= ratio <= 2.0


class GatedTracker:
    def __init__(
        self,
        acq_conf: float = 0.55,
        acq_frames: int = 4,
        acq_radius: float = 80.0,
        acq_max_gap: int = 2,
        app_min: float = 0.30,
        match_min: float = 0.40,
        w_app: float = 0.6,
        w_spatial: float = 0.4,
        template_ema: float = 0.04,
        template_size: int = 24,
        miss_budget: int = 15,      
    ):
        self.acq_conf = acq_conf
        self.acq_frames = acq_frames
        self.acq_radius = acq_radius
        self.acq_max_gap = acq_max_gap
        self.app_min = app_min
        self.match_min = match_min
        self.w_app = w_app
        self.w_spatial = w_spatial
        self.template_ema = template_ema
        self.template_size = template_size
        self.miss_budget = miss_budget

        self.lk = LKSingleTargetTracker(
            estimate_scale=False,
            template_size=template_size,
            max_consecutive_misses=10 ** 9,
        )

        self.anchor_sig: Optional[np.ndarray] = None
        self.running_sig: Optional[np.ndarray] = None
        self.miss = 0
        self._acq: list[dict] = []                      # pending acquisition candidates

    def _patch(self, gray, bbox):
        x1, y1, x2, y2 = bbox
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i = min(gray.shape[1], int(x2))
        y2i = min(gray.shape[0], int(y2))
        if x2i - x1i < 4 or y2i - y1i < 4:
            return None
        crop = gray[y1i:y2i, x1i:x2i]
        s = self.template_size
        p = cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
        p -= p.mean()
        sd = float(p.std())
        if sd < 1e-6:
            return None
        return p / sd

    def _appearance(self, gray, bbox) -> float:
        p = self._patch(gray, bbox)
        if p is None:
            return -1.0
        scores = []
        if self.anchor_sig is not None:
            scores.append(float((self.anchor_sig * p).mean()))
        if self.running_sig is not None:
            scores.append(float((self.running_sig * p).mean()))
        return max(scores) if scores else 1.0

    def _update_running_template(self, gray, bbox):
        p = self._patch(gray, bbox)
        if p is None:
            return
        if self.running_sig is None:
            self.running_sig = p
            return
        blended = (1.0 - self.template_ema) * self.running_sig + self.template_ema * p
        blended -= blended.mean()
        sd = float(blended.std())
        if sd > 1e-6:
            self.running_sig = blended / sd

    def step(self, prev_gray, curr_gray, detections) -> TrackOutput:
        if self.lk.state == LKState.locked:
            return self._step_locked(prev_gray, curr_gray, detections)
        return self._step_acquire(curr_gray, detections)

    def _step_locked(self, prev_gray, curr_gray, detections) -> TrackOutput:
        if prev_gray is not None:
            self.lk.update(prev_gray, curr_gray)
        predicted = self.lk.bbox

        best, best_match, best_app = None, -1e9, -1.0
        for dbox, _conf in detections:
            app = self._appearance(curr_gray, dbox)
            if app < self.app_min:                       # appearance floor
                continue
            if not self.lk.size_consistent(dbox):        # size gate
                continue
            spatial = _iou(dbox, predicted) if predicted is not None else 0.0
            match = self.w_app * app + self.w_spatial * spatial
            if match > best_match:
                best, best_match, best_app = dbox, match, app

        if best is not None and best_match >= self.match_min:
            self.lk.update_from_detection(curr_gray, best)
            self._update_running_template(curr_gray, best)
            self.miss = 0
            return TrackOutput("LOCKED", self.lk.bbox, "DET", best_app, best_match)

        self.miss += 1
        if self.miss >= self.miss_budget or self.lk.bbox is None:
            last = self.lk.bbox
            self.lk._reset(keep_signature=True)          # keep size stats
            self.running_sig = None                      # anchor kept for re-acq
            self.miss = 0
            self._acq.clear()
            return TrackOutput("LOST", last, "LOST", 0.0, 0.0)
        return TrackOutput("LOCKED", self.lk.bbox, "LK", 0.0, 0.0)

    def _step_acquire(self, curr_gray, detections) -> TrackOutput:
        # candidates must clear the acquisition confidence; when re-acquiring a known target (anchor set) they must also look like it.
        strong = []
        for dbox, conf in detections:
            if conf < self.acq_conf:
                continue
            if self.anchor_sig is not None and \
                    self._appearance(curr_gray, dbox) < self.app_min:
                continue
            strong.append((dbox, conf))

        # age every pending candidate; matched ones get reset to gap 0.
        for c in self._acq:
            c["gap"] += 1
        for dbox, conf in strong:
            cx, cy = _center(dbox)
            match = None
            for c in self._acq:
                ccx, ccy = _center(c["bbox"])
                if (ccx - cx) ** 2 + (ccy - cy) ** 2 <= self.acq_radius ** 2:
                    match = c
                    break
            if match is not None:
                match["bbox"] = dbox
                match["conf"] = conf
                match["hits"] += 1
                match["gap"] = 0
            else:
                self._acq.append({"bbox": dbox, "conf": conf, "hits": 1, "gap": 0})

        # drop candidates unseen for too long (a one-frame bird blip dies here).
        self._acq = [c for c in self._acq if c["gap"] <= self.acq_max_gap]

        # promote the most-persistent candidate once it clears K hits.
        ready = [c for c in self._acq if c["hits"] >= self.acq_frames]
        if ready:
            winner = max(ready, key=lambda c: (c["hits"], c["conf"]))
            if self.lk.update_from_detection(curr_gray, winner["bbox"]):
                # capture the fixed anchor template at the moment of lock.
                anchor = self._patch(curr_gray, winner["bbox"])
                if anchor is not None:
                    self.anchor_sig = anchor
                    self.running_sig = anchor.copy()
                self._acq.clear()
                self.miss = 0
                return TrackOutput("LOCKED", self.lk.bbox, "LOCK", 1.0, 1.0)

        return TrackOutput("SEARCH", None, "SEARCH", 0.0, 0.0)


def open_camera(index, width, height):
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index} - try --camera 1")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_fov(path):
    f = np.load(path)
    return (f["map_x"], f["map_y"], float(f["alpha"]),
            float(f["beta"]), int(f["imgsz"]))


_COLORS = {"LOCK": (0, 255, 255), "DET": (0, 255, 0),
           "LK": (0, 165, 255), "LOST": (0, 0, 255), "SEARCH": (160, 160, 160)}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hef", default="telescope_lvl2.hef", help="path to the compiled .hef")
    p.add_argument("--fov", default=None,help="foveation .npz (telescope HEFs only; omit for plain v8/v11)")
    p.add_argument("--camera", type=int, default=0, help="USB camera index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf", type=float, default=CONF, help="detector score floor")
    p.add_argument("--iou", type=float, default=IOU, help="NMS IoU threshold")
    p.add_argument("--acq-conf", type=float, default=0.55,help="detector conf for a region to be an acquisition candidate")
    p.add_argument("--acq-frames", type=int, default=4,help="K: frames a candidate must persist before the first lock")
    p.add_argument("--app-min", type=float, default=0.30,help="min appearance NCC for a detection to be considered")
    p.add_argument("--match-min", type=float, default=0.40,help="min combined (appearance+spatial) score to accept a det")
    p.add_argument("--miss-budget", type=int, default=15,help="consecutive unconfirmed frames before declaring LOST")

    p.add_argument("--no-show", action="store_true", help="headless")
    p.add_argument("--out-dir", default="recordings")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.hef).exists():
        raise FileNotFoundError(f"HEF not found: {args.hef}")
    fov = None
    if args.fov:
        if not Path(args.fov).exists():
            raise FileNotFoundError(f"foveation LUT not found: {args.fov}")
        fov = load_fov(args.fov)
        print(f"foveation: alpha={fov[2]:.4f} beta={fov[3]:.4f} imgsz={fov[4]}")

    print(f"loading HEF: {args.hef}")
    model = HailoYOLOv8(args.hef)
    tracker = GatedTracker(
        acq_conf=args.acq_conf, acq_frames=args.acq_frames,
        app_min=args.app_min, match_min=args.match_min,
        miss_budget=args.miss_budget,
    )

    cap = open_camera(args.camera, args.width, args.height)
    print("Camera opened. Press 'q' to quit." if not args.no_show
          else "Camera opened (headless). Ctrl+C to quit.")

    writer, out_path = None, None
    if not args.no_record:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"tracking_{stamp}.mp4"

    prev_gray = None
    n, infer_ms_s = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break
            if args.max_frames and n >= args.max_frames:
                break
            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if writer is None and out_path is not None:
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (W, H))
                if not writer.isOpened():
                    print(f"could not open VideoWriter {out_path} - recording off")
                    writer, out_path = None, None
                else:
                    print(f"Recording to: {out_path}")

            detections, infer_ms = run_detector(model, frame, fov, args.conf, args.iou)
            infer_ms_s = infer_ms if n == 0 else 0.9 * infer_ms_s + 0.1 * infer_ms

            out = tracker.step(prev_gray, gray, detections)

            for dbox, conf in detections:
                x1, y1, x2, y2 = (int(v) for v in dbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
            if out.bbox is not None and out.state != "LOST":
                x1, y1, x2, y2 = (int(v) for v in out.bbox)
                color = _COLORS.get(out.driver, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = out.driver
                if out.driver == "DET":
                    tag = f"DET ncc={out.appearance:.2f}"
                cv2.putText(frame, tag, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            infer_fps = 1000.0 / max(infer_ms_s, 1e-6)
            cv2.putText(frame,
                        f"{out.state}  |  YOLO {infer_fps:5.1f} FPS "
                        f"({infer_ms_s:4.1f}ms)  |  {len(detections)} det",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if writer is not None:
                writer.write(frame)
            if args.no_show:
                if out.state != "SEARCH":
                    print(f"f{n:05d} {out.state:<7} YOLO {infer_fps:5.1f}fps "
                          f"drv={out.driver:<6} ncc={out.appearance:.2f} "
                          f"match={out.match:.2f}")
            else:
                cv2.imshow("appearance-gated tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_gray = gray
            n += 1
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        model.close()
        if not args.no_show:
            cv2.destroyAllWindows()
        print(f"done -- {n} frames")
        if out_path is not None:
            print(f"saved: {out_path}")


if __name__ == "__main__":
    main()