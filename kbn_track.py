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

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,ConfigureParams, InputVStreamParams,OutputVStreamParams, FormatType)
    _HAS_HAILO = True
except ImportError:
    _HAS_HAILO = False
    HEF = VDevice = HailoStreamInterface = InferVStreams = None
    ConfigureParams = InputVStreamParams = OutputVStreamParams = FormatType = None


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
        if not _HAS_HAILO:
            raise RuntimeError(
                "hailo_platform is not installed. "
                "Use test_track.py for offline testing with a .pt model.")
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


def _bbox_to_z(bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5,x2 - x1, y2 - y1], dtype=np.float64)


def _z_to_bbox(z) -> tuple:
    cx, cy, w, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    w = max(1.0, w); h = max(1.0, h)
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class LKMotionMeasurer:
    def __init__(self):
        self._pts: Optional[np.ndarray] = None

    def seed(self, gray, bbox) -> bool:
        x1, y1, x2, y2 = bbox
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i = min(gray.shape[1], int(x2))
        y2i = min(gray.shape[0], int(y2))
        if x2i - x1i < 6 or y2i - y1i < 6:
            self._pts = None
            return False
        roi = gray[y1i:y2i, x1i:x2i]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=80, qualityLevel=0.01,minDistance=4)
        if pts is None or len(pts) < 6:
            # texture-poor crop: regular 6x6 grid keeps LK working on plain fuselage.
            gx = np.linspace(x1i + 2, x2i - 2, 6)
            gy = np.linspace(y1i + 2, y2i - 2, 6)
            mx, my = np.meshgrid(gx, gy)
            grid = np.stack([mx.ravel(), my.ravel()], axis=1)
            self._pts = grid.astype(np.float32).reshape(-1, 1, 2)
            return True
        pts = pts.reshape(-1, 2)
        pts[:, 0] += x1i
        pts[:, 1] += y1i
        self._pts = pts.astype(np.float32).reshape(-1, 1, 2)
        return True

    def measure(self, prev_gray, curr_gray) -> Optional[tuple]:
        if self._pts is None or prev_gray is None:
            return None
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, self._pts, None, **_LK_PARAMS)
        if nxt is None or st is None:
            return None
        ok = st.ravel() == 1
        good_old = self._pts[ok].reshape(-1, 2)
        good_new = nxt[ok].reshape(-1, 2)
        if len(good_new) < 4:
            return None
        disp = good_new - good_old
        dx = float(np.median(disp[:, 0]))
        dy = float(np.median(disp[:, 1]))
        # carry forward the inliers as the new seed (drops drifting points).
        self._pts = good_new.astype(np.float32).reshape(-1, 1, 2)
        return (dx, dy, int(len(good_new)))


class KalmanBox:
    _DIM = 8
    _OBS_BOX = 4
    _OBS_VEL = 2

    def __init__(self, bbox):
        z = _bbox_to_z(bbox)
        self.x = np.zeros(self._DIM, dtype=np.float64)
        self.x[:4] = z
        # initial covariance: tight on position, loose on velocity.
        self.P = np.diag([16.0, 16.0, 64.0, 64.0,400.0, 400.0, 100.0, 100.0]).astype(np.float64)
        # process noise scaled per-axis. Velocity noise dominates so the
        # filter can absorb sudden manoeuvres without huge position errors.
        self._Q = np.diag([1.0, 1.0, 2.0, 2.0,25.0, 25.0, 4.0, 4.0]).astype(np.float64)
        # measurement noise for full bbox observations.
        self._R_box = np.diag([4.0, 4.0, 16.0, 16.0]).astype(np.float64)
        # measurement noise for LK velocity observations.
        self._R_vel = np.diag([9.0, 9.0]).astype(np.float64)
        # H matrices precomputed.
        self._H_box = np.zeros((self._OBS_BOX, self._DIM), dtype=np.float64)
        for i in range(4):
            self._H_box[i, i] = 1.0
        self._H_vel = np.zeros((self._OBS_VEL, self._DIM), dtype=np.float64)
        self._H_vel[0, 4] = 1.0
        self._H_vel[1, 5] = 1.0

    def predict(self, dt: float = 1.0):
        F = np.eye(self._DIM, dtype=np.float64)
        F[0, 4] = dt; F[1, 5] = dt; F[2, 6] = dt; F[3, 7] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q * dt

    def update_bbox(self, bbox, r_scale: float = 1.0):
        z = _bbox_to_z(bbox)
        self._update(z, self._H_box, self._R_box * r_scale)

    def update_velocity(self, dx: float, dy: float, r_scale: float = 1.0):
        z = np.array([dx, dy], dtype=np.float64)
        self._update(z, self._H_vel, self._R_vel * r_scale)

    def _update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        I = np.eye(self._DIM, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def bbox(self) -> tuple:
        return _z_to_bbox(self.x[:4])

    def mahalanobis2(self, bbox) -> float:
        z = _bbox_to_z(bbox)
        y = z - self._H_box @ self.x
        S = self._H_box @ self.P @ self._H_box.T + self._R_box
        try:
            return float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return float("inf")


_INFEASIBLE = 1.0e6  # use a large finite number, not inf, for safety with LSA


def _assign(cost: np.ndarray, infeasible: float = _INFEASIBLE / 2.0):
    if cost.size == 0:
        return []
    if _HAS_SCIPY:
        rows, cols = _scipy_lsa(cost)
        pairs = []
        for r, c in zip(rows, cols):
            if float(cost[r, c]) < infeasible:
                pairs.append((int(r), int(c)))
        return pairs
    # greedy fallback
    flat = [(float(cost[r, c]), r, c)
            for r in range(cost.shape[0])
            for c in range(cost.shape[1])
            if float(cost[r, c]) < infeasible]
    flat.sort(key=lambda x: x[0])
    used_r, used_c, pairs = set(), set(), []
    for _v, r, c in flat:
        if r in used_r or c in used_c:
            continue
        used_r.add(r); used_c.add(c)
        pairs.append((r, c))
    return pairs


# Chi-squared 0.99 critical values for the Mahalanobis bbox gate (df=4).
_MAH_GATE_TIGHT = 13.28
_MAH_GATE_LOOSE = 23.21   # df=4, p=0.9999 - looser pass-2 gate


class _Track:
    def __init__(self, track_id: int, gray, bbox, conf: float,
                 template_size: int):
        self.id = track_id
        self.kf = KalmanBox(bbox)
        self.lk = LKMotionMeasurer()
        self.lk.seed(gray, bbox)
        self.template_size = template_size
        self.anchor_sig: Optional[np.ndarray] = None
        self.running_sig: Optional[np.ndarray] = None
        # reference size for the hard size-band safety check.
        self.ref_size = (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
        self.hits = 1
        self.misses = 0
        self.gap = 0
        self.confirmed = False
        self.last_conf = conf
        self.last_app = 0.0
        self.last_match = 0.0
        self.driver = "LOCK"

    @property
    def bbox(self):
        return self.kf.bbox()

    def predict(self, prev_gray, curr_gray):
        self.kf.predict()
        if prev_gray is not None:
            disp = self.lk.measure(prev_gray, curr_gray)
            if disp is not None:
                dx, dy, _n = disp
                self.kf.update_velocity(dx, dy)

    def patch(self, gray, bbox):
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

    def appearance(self, gray, bbox) -> float:
        p = self.patch(gray, bbox)
        if p is None:
            return -1.0
        scores = []
        if self.anchor_sig is not None:
            scores.append(float((self.anchor_sig * p).mean()))
        if self.running_sig is not None:
            scores.append(float((self.running_sig * p).mean()))
        return max(scores) if scores else 1.0

    def size_consistent(self, bbox) -> bool:
        rw, rh = self.ref_size
        if rw <= 1.0 or rh <= 1.0:
            return True
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        ratio = float(np.sqrt((w * h) / (rw * rh)))
        return 0.5 <= ratio <= 2.0

    def bind_detection(self, gray, bbox, conf: float, app: float,match: float, template_ema: float, r_scale: float = 1.0):
        self.kf.update_bbox(bbox, r_scale=r_scale)
        self.lk.seed(gray, self.kf.bbox())  # re-seed at filtered position
        self.ref_size = (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
        self.last_conf = conf
        self.last_app = app
        self.last_match = match
        self.hits += 1
        self.misses = 0
        self.gap = 0
        p = self.patch(gray, bbox)
        if p is None:
            return
        if self.running_sig is None:
            self.running_sig = p
            return
        blended = (1.0 - template_ema) * self.running_sig + template_ema * p
        blended -= blended.mean()
        sd = float(blended.std())
        if sd > 1e-6:
            self.running_sig = blended / sd

    def confirm(self, gray):
        p = self.patch(gray, self.kf.bbox())
        if p is not None:
            self.anchor_sig = p
            if self.running_sig is None:
                self.running_sig = p.copy()
        self.confirmed = True
        self.last_app = 1.0
        self.last_match = 1.0

class GatedTracker2:
    def __init__(
        self,
        acq_conf: float = 0.55,
        low_conf: float = 0.10,         # pass-2 floor
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
        reid_app_min: float = 0.45,
        max_tracks: int = 8,
        gate_tight: float = _MAH_GATE_TIGHT,
        gate_loose: float = _MAH_GATE_LOOSE,
        pass2_iou_min: float = 0.35,
    ):
        self.acq_conf = acq_conf
        self.low_conf = low_conf
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
        self.reid_app_min = reid_app_min
        self.max_tracks = max_tracks
        self.gate_tight = gate_tight
        self.gate_loose = gate_loose
        self.pass2_iou_min = pass2_iou_min

        self.tracks: list[_Track] = []
        self.primary_id: Optional[int] = None
        self.dead_anchor: Optional[np.ndarray] = None
        self._next_id = 1

    
    def _spawn(self, gray, bbox, conf) -> Optional[_Track]:
        if len(self.tracks) >= self.max_tracks:
            return None
        t = _Track(self._next_id, gray, bbox, conf, self.template_size)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _pass1_cost(self, track: _Track, gray, dbox):
        m2 = track.kf.mahalanobis2(dbox)
        if m2 > self.gate_tight:
            return _INFEASIBLE, 0.0, 0.0
        if not track.size_consistent(dbox):
            return _INFEASIBLE, 0.0, 0.0
        iou_s = _iou(dbox, track.bbox)
        if track.confirmed:
            app = track.appearance(gray, dbox)
            if app < self.app_min:
                return _INFEASIBLE, app, 0.0
            match = self.w_app * app + self.w_spatial * iou_s
            if match < self.match_min:
                return _INFEASIBLE, app, match
            return 1.0 - match, app, match
        match = iou_s
        return 1.0 - match, 0.0, match

    def _pass2_cost(self, track: _Track, dbox):
        m2 = track.kf.mahalanobis2(dbox)
        if m2 > self.gate_loose:
            return _INFEASIBLE, 0.0
        if not track.size_consistent(dbox):
            return _INFEASIBLE, 0.0
        iou_s = _iou(dbox, track.bbox)
        if iou_s < self.pass2_iou_min:
            return _INFEASIBLE, 0.0
        return 1.0 - iou_s, iou_s

    def _get(self, track_id: Optional[int]) -> Optional[_Track]:
        if track_id is None:
            return None
        for t in self.tracks:
            if t.id == track_id:
                return t
        return None

    def _pick_primary(self, gray) -> Optional[_Track]:
        confirmed = [t for t in self.tracks if t.confirmed]
        if not confirmed:
            return None
        if self.dead_anchor is not None:
            best, best_score = None, -1.0
            for t in confirmed:
                if t.bbox is None:
                    continue
                p = t.patch(gray, t.bbox)
                if p is None:
                    continue
                s = float((self.dead_anchor * p).mean())
                if s > best_score:
                    best, best_score = t, s
            if best is not None and best_score >= self.reid_app_min:
                best.anchor_sig = self.dead_anchor
                self.dead_anchor = None
                return best
            # waiting for the original - don't promote an unrelated track.
            return None
        return max(confirmed, key=lambda t: (t.hits, t.last_match))

    def step(self, prev_gray, curr_gray, detections) -> TrackOutput:
        # predict every track. LK velocity fed in here too.
        for t in self.tracks:
            t.predict(prev_gray, curr_gray)

        # split detections by confidence band.
        high_idx = [j for j, (_, c) in enumerate(detections) if c >= self.acq_conf]
        low_idx  = [j for j, (_, c) in enumerate(detections)
                    if self.low_conf <= c < self.acq_conf]

        #high-conf detections vs every track 
        track_idx = list(range(len(self.tracks)))
        pairs1 = self._associate(track_idx, high_idx, curr_gray,
                                 detections, pass2=False)

        # bind pass-1 matches.
        matched_t = {r for r, _ in pairs1}
        matched_d = {c for _, c in pairs1}
        for r, c in pairs1:
            t = self.tracks[r]
            dbox, dconf = detections[c]
            app, match = self._cost_lookup_pass1(t, curr_gray, dbox)
            was_confirmed = t.confirmed
            t.bind_detection(curr_gray, dbox, dconf, app, match,
                             self.template_ema)
            t.driver = "DET" if was_confirmed else "LOCK"

        # low-conf detections vs unmatched confirmed tracks ----
        unmatched_conf = [i for i, t in enumerate(self.tracks)
                          if t.confirmed and i not in matched_t]
        pairs2 = self._associate(unmatched_conf, low_idx, curr_gray,
                                 detections, pass2=True)
        for r, c in pairs2:
            t = self.tracks[r]
            dbox, dconf = detections[c]
            iou_s = _iou(dbox, t.bbox)
            # BYTE update: trust position but inflate measurement noise since
            # low-confidence detections are typically less accurate.
            t.bind_detection(curr_gray, dbox, dconf, 0.0, iou_s,
                             self.template_ema, r_scale=2.0)
            t.driver = "BYTE"
            matched_t.add(r)
            matched_d.add(c)

        #  coast unmatched tracks 
        for i, t in enumerate(self.tracks):
            if i in matched_t:
                continue
            if t.confirmed:
                t.misses += 1
                t.driver = "LK"
                t.last_app = 0.0
                t.last_match = 0.0
            else:
                t.gap += 1

        # spawn tentatives from leftover strong detections 
        for j, (dbox, dconf) in enumerate(detections):
            if j in matched_d or dconf < self.acq_conf:
                continue
            self._spawn(curr_gray, dbox, dconf)

        # promote tentatives 
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.acq_frames:
                t.confirm(curr_gray)
                t.driver = "LOCK"

        #  cull dead track
        primary_died = False
        last_primary_bbox = None
        alive = []
        for t in self.tracks:
            if t.confirmed:
                if t.misses >= self.miss_budget:
                    if t.id == self.primary_id:
                        primary_died = True
                        last_primary_bbox = t.bbox
                        self.dead_anchor = t.anchor_sig
                    continue
            else:
                if t.gap > self.acq_max_gap:
                    continue
            alive.append(t)
        self.tracks = alive

        # primary maintenance
        primary = self._get(self.primary_id)
        if primary is None:
            primary = self._pick_primary(curr_gray)
            self.primary_id = primary.id if primary is not None else None

        if primary is None:
            if primary_died:
                return TrackOutput("LOST", last_primary_bbox, "LOST", 0.0, 0.0)
            return TrackOutput("SEARCH", None, "SEARCH", 0.0, 0.0)
        return TrackOutput("LOCKED", primary.bbox, primary.driver,
                           primary.last_app, primary.last_match)

    def _associate(self, track_indices, det_indices, gray, detections,pass2: bool):
        n_t, n_d = len(track_indices), len(det_indices)
        if n_t == 0 or n_d == 0:
            return []
        cost = np.full((n_t, n_d), _INFEASIBLE, dtype=np.float64)
        for i, ti in enumerate(track_indices):
            t = self.tracks[ti]
            for j, dj in enumerate(det_indices):
                dbox, _dconf = detections[dj]
                if pass2:
                    c, _iou_s = self._pass2_cost(t, dbox)
                else:
                    c, _app, _match = self._pass1_cost(t, gray, dbox)
                cost[i, j] = c
        local_pairs = _assign(cost)
        return [(track_indices[i], det_indices[j]) for (i, j) in local_pairs]

    def _cost_lookup_pass1(self, t: _Track, gray, dbox):
        _c, app, match = self._pass1_cost(t, gray, dbox)
        return app, match


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
           "LK": (0, 165, 255), "BYTE": (100, 255, 100),
           "LOST": (0, 0, 255), "SEARCH": (160, 160, 160)}

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hef", default="best.hef", help="path to the compiled .hef")
    p.add_argument("--fov", default="telescope_foveation.npz",
                   help="foveation .npz - required for telescope HEFs (the "
                        "default). Pass --fov '' for a plain v8/v11 HEF.")
    p.add_argument("--camera", type=int, default=0, help="USB camera index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf", type=float, default=CONF, help="detector score floor")
    p.add_argument("--iou", type=float, default=IOU, help="NMS IoU threshold")
    # gating knobs
    p.add_argument("--acq-conf", type=float, default=0.55,
                   help="min conf for pass-1 (also spawn threshold)")
    p.add_argument("--low-conf", type=float, default=0.10,
                   help="pass-2 low-confidence floor (ByteTrack recovery)")
    p.add_argument("--acq-frames", type=int, default=4,
                   help="K: frames a candidate must persist before the first lock")
    p.add_argument("--app-min", type=float, default=0.30,
                   help="min appearance NCC for a detection to be considered")
    p.add_argument("--match-min", type=float, default=0.40,
                   help="min combined (appearance+spatial) score to accept a det")
    p.add_argument("--miss-budget", type=int, default=15,
                   help="consecutive unconfirmed frames before declaring LOST")
    p.add_argument("--reid-app-min", type=float, default=0.45,
                   help="min NCC against dead anchor to reacquire primary")
    p.add_argument("--max-tracks", type=int, default=8)
    # output
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
    if args.fov and args.fov.strip():
        if not Path(args.fov).exists():
            raise FileNotFoundError(f"foveation LUT not found: {args.fov}")
        fov = load_fov(args.fov)
        print(f"foveation: alpha={fov[2]:.4f} beta={fov[3]:.4f} imgsz={fov[4]}")
    else:
        print("foveation: disabled - make sure your HEF was trained without it")

    print(f"loading HEF: {args.hef}")
    if not _HAS_SCIPY:
        print("note: scipy not found - falling back to greedy assignment "
              "(apt install python3-scipy for Hungarian)")
    model = HailoYOLOv8(args.hef)
    tracker = GatedTracker2(
        acq_conf=args.acq_conf, low_conf=args.low_conf,
        acq_frames=args.acq_frames,
        app_min=args.app_min, match_min=args.match_min,
        miss_budget=args.miss_budget,
        reid_app_min=args.reid_app_min,
        max_tracks=args.max_tracks,
    )

    cap = open_camera(args.camera, args.width, args.height)
    print("Camera opened. Press 'q' to quit." if not args.no_show
          else "Camera opened (headless). Ctrl+C to quit.")

    writer, out_path = None, None
    if not args.no_record:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"tracking2_{stamp}.mp4"

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
                elif out.driver == "BYTE":
                    tag = f"BYTE iou={out.match:.2f}"
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
                cv2.imshow("track2: kalman + bytetrack", frame)
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
