from __future__ import annotations

import argparse
import contextlib
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,InferVStreams, ConfigureParams,InputVStreamParams, OutputVStreamParams,FormatType)
    _HAS_HAILO = True
except ImportError:
    _HAS_HAILO = False
    HEF = VDevice = HailoStreamInterface = InferVStreams = None
    ConfigureParams = InputVStreamParams = OutputVStreamParams = FormatType = None

# Vendored ByteTrack (defined below) needs only numpy. scipy gives optimal
# (Hungarian) assignment; without it we fall back to greedy matching. NO
# ultralytics / torch import -> runs on a Hailo Pi with just numpy + opencv +
# hailo_platform (and optionally scipy).
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    _scipy_lsa = None

MODEL_SIZE = 640
DFL_BINS   = 16
CONF_FLOOR = 0.25
IOU_NMS    = 0.45
_EPS       = 1e-6


C_DET    = (  0, 255,   0)   # green   -- tracker driven by a fresh detection
C_LK     = (  0, 165, 255)   # orange  -- tracker coasted by LK (short gap)
C_LOST   = (  0,   0, 255)   # red     -- track lost (id freed)
C_SEARCH = (255, 100, 255)   # magenta -- searching for re-acquisition
C_RAW    = (255, 255,   0)   # cyan    -- raw detector boxes (thin)
C_TEXT_BG = (  0,   0,   0)  # black box behind labels for legibility
C_TEXT   = (255, 255, 255)


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
        xs = np.concatenate([x1 + t * (x2 - x1), x1 + t * (x2 - x1),np.full(n_edge, x1), np.full(n_edge, x2)])
        ys = np.concatenate([np.full(n_edge, y1), np.full(n_edge, y2),y1 + t * (y2 - y1), y1 + t * (y2 - y1)])
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
            raise RuntimeError("hailo_platform not installed")
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

    def infer(self, rgb_uint8):
        return self.pipe.infer({self.in_info.name: rgb_uint8[np.newaxis, ...]})

    def close(self):
        self._stack.close()


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
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=80,
                                      qualityLevel=0.01, minDistance=4)
        if pts is None or len(pts) < 6:
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

    def measure(self, prev_gray, curr_gray):
        if self._pts is None or prev_gray is None:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
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
        old_sd = good_old.std(axis=0).mean() + _EPS
        new_sd = good_new.std(axis=0).mean() + _EPS
        scale = float(np.clip(new_sd / old_sd, 0.85, 1.15))   # gentle bound
        self._pts = good_new.astype(np.float32).reshape(-1, 1, 2)
        return (dx, dy, scale)

def run_detector(model, frame, fov, conf, iou):
    H, W = frame.shape[:2]
    lb, scale, padx, pady = letterbox(frame, MODEL_SIZE)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    net_in = rgb
    if fov is not None:
        map_x, map_y, _, _, _ = fov
        net_in = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
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


class _DetResults:
    __slots__ = ("conf", "xywh", "xyxy", "cls")

    def __init__(self, conf, xywh, xyxy, cls):
        self.conf = conf
        self.xywh = xywh
        self.xyxy = xyxy
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _DetResults(self.conf[idx], self.xywh[idx],
                           self.xyxy[idx], self.cls[idx])


def _wrap_results(detections):
    n = len(detections)
    if n == 0:
        return _DetResults(
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    xyxy = np.array([d[0] for d in detections], dtype=np.float32)
    conf = np.array([d[1] for d in detections], dtype=np.float32)
    xywh = np.empty_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    xywh[:, 2] =  xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] =  xyxy[:, 3] - xyxy[:, 1]
    cls = np.zeros(n, dtype=np.float32)
    return _DetResults(conf, xywh, xyxy, cls)


def _make_bytetrack_args(track_buffer, match_thresh, high, low, new_thr):
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=high,
        track_low_thresh=low,
        new_track_thresh=new_thr,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=True,
    )


def _make_botsort_args(track_buffer, match_thresh, high, low, new_thr):
    return SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=high,
        track_low_thresh=low,
        new_track_thresh=new_thr,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=True,
        gmc_method="sparseOptFlow",     # camera-motion compensation
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,                # no ReID model -> appearance disabled
    )


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


# ----------------------------------------------------------------------
# Vendored, torch-free ByteTrack (numpy only; optional scipy assignment).
# Faithful port of the standard ByteTrack: 8-state constant-velocity Kalman,
# high/low two-pass IoU association, lost-track buffer, unconfirmed handling.
# Replaces ultralytics.trackers.BYTETracker so no torch is needed on the Pi.
# `update()` returns rows [x1, y1, x2, y2, track_id, score, cls, idx].
# ----------------------------------------------------------------------
def _ious(atlbr, btlbr):
    A, B = len(atlbr), len(btlbr)
    out = np.zeros((A, B), dtype=np.float32)
    if A == 0 or B == 0:
        return out
    a = np.asarray(atlbr, dtype=np.float32)
    b = np.asarray(btlbr, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(A):
        ix1 = np.maximum(a[i, 0], b[:, 0]); iy1 = np.maximum(a[i, 1], b[:, 1])
        ix2 = np.minimum(a[i, 2], b[:, 2]); iy2 = np.minimum(a[i, 3], b[:, 3])
        iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        out[i] = inter / (area_a[i] + area_b - inter + 1e-9)
    return out


def _iou_distance(tracks, dets):
    return 1.0 - _ious([t.tlbr for t in tracks], [d.tlbr for d in dets])


def _linear_assignment(cost, thresh):
    """Return (matches Nx2, unmatched_a, unmatched_b). cost = 1 - IoU."""
    if cost.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost.shape[0])), tuple(range(cost.shape[1])))
    matches = []
    if _HAS_SCIPY:
        rows, cols = _scipy_lsa(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= thresh:
                matches.append((r, c))
    else:  # greedy fallback: take lowest-cost pairs first
        order = sorted(((cost[r, c], r, c)
                        for r in range(cost.shape[0])
                        for c in range(cost.shape[1])), key=lambda z: z[0])
        ua, ub = set(range(cost.shape[0])), set(range(cost.shape[1]))
        for cst, r, c in order:
            if cst > thresh:
                break
            if r in ua and c in ub:
                matches.append((r, c)); ua.discard(r); ub.discard(c)
    ma = {r for r, _ in matches}
    mb = {c for _, c in matches}
    un_a = tuple(r for r in range(cost.shape[0]) if r not in ma)
    un_b = tuple(c for c in range(cost.shape[1]) if c not in mb)
    return np.array(matches, dtype=int).reshape(-1, 2), un_a, un_b


class _KalmanXYAH:
    """8-state (x,y,a,h,vx,vy,va,vh) constant-velocity KF; measures (x,y,a,h)."""
    def __init__(self):
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, 4 + i] = 1.0
        self._H = np.eye(4, 8)
        self._spw = 1.0 / 20
        self._svw = 1.0 / 160

    def initiate(self, m):
        mean = np.r_[m, np.zeros(4)]
        std = [2*self._spw*m[3], 2*self._spw*m[3], 1e-2, 2*self._spw*m[3],
               10*self._svw*m[3], 10*self._svw*m[3], 1e-5, 10*self._svw*m[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        std = [self._spw*mean[3], self._spw*mean[3], 1e-2, self._spw*mean[3],
               self._svw*mean[3], self._svw*mean[3], 1e-5, self._svw*mean[3]]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def project(self, mean, cov):
        std = [self._spw*mean[3], self._spw*mean[3], 1e-1, self._spw*mean[3]]
        R = np.diag(np.square(std))
        return self._H @ mean, self._H @ cov @ self._H.T + R

    def update(self, mean, cov, meas):
        pm, pc = self.project(mean, cov)
        B = cov @ self._H.T                      # 8x4
        K = np.linalg.solve(pc, B.T).T           # 8x4 Kalman gain
        new_mean = mean + K @ (meas - pm)
        new_cov = cov - K @ pc @ K.T
        return new_mean, new_cov


class _STrack:
    _kf = _KalmanXYAH()
    _count = 0

    def __init__(self, tlwh, score, cls):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.mean = self.cov = None
        self.is_activated = False
        self.score = float(score)
        self.cls = float(cls)
        self.track_id = 0
        self.state = "new"
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @staticmethod
    def _xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def next_id():
        _STrack._count += 1
        return _STrack._count

    def predict(self):
        if self.mean is None:
            return
        mean = self.mean.copy()
        if self.state != "tracked":
            mean[7] = 0.0
        self.mean, self.cov = self._kf.predict(mean, self.cov)

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.mean, self.cov = self._kf.initiate(self._xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = "tracked"
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new, frame_id, new_id=False):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, self._xyah(new.tlwh))
        self.tracklet_len = 0
        self.state = "tracked"
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score, self.cls = new.score, new.cls

    def update(self, new, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.cov = self._kf.update(self.mean, self.cov, self._xyah(new.tlwh))
        self.state = "tracked"
        self.is_activated = True
        self.score, self.cls = new.score, new.cls

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"


class _ByteTracker:
    """Drop-in for ultralytics BYTETracker. update(results, img=None) reads
    results.conf / results.xywh (centre xywh) / results.cls."""
    def __init__(self, args, frame_rate=30):
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        _STrack._count = 0

    @staticmethod
    def _join(a, b):
        seen = {t.track_id for t in a}
        res = list(a)
        for t in b:
            if t.track_id not in seen:
                seen.add(t.track_id); res.append(t)
        return res

    @staticmethod
    def _sub(a, b):
        ids = {t.track_id for t in b}
        return [t for t in a if t.track_id not in ids]

    @staticmethod
    def _remove_dup(a, b):
        d = _iou_distance(a, b)
        dupa, dupb = set(), set()
        for p, q in zip(*np.where(d < 0.15)):
            tp = a[p].frame_id - a[p].start_frame
            tq = b[q].frame_id - b[q].start_frame
            (dupb if tp > tq else dupa).add(q if tp > tq else p)
        ra = [t for i, t in enumerate(a) if i not in dupa]
        rb = [t for i, t in enumerate(b) if i not in dupb]
        return ra, rb

    def update(self, results, img=None):
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        n = len(results)
        scores = np.asarray(results.conf, dtype=np.float32) if n else np.zeros(0, np.float32)
        bboxes = np.asarray(results.xywh, dtype=np.float32) if n else np.zeros((0, 4), np.float32)
        clses = np.asarray(results.cls, dtype=np.float32) if n else np.zeros(0, np.float32)

        remain = scores >= self.args.track_high_thresh
        second = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        def make(mask):
            out = []
            for i in np.where(mask)[0]:
                cx, cy, w, h = bboxes[i]
                out.append(_STrack([cx - w / 2, cy - h / 2, w, h], scores[i], clses[i]))
            return out

        dets = make(remain)
        dets_second = make(second)

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        pool = self._join(tracked, self.lost)
        for t in pool:
            t.predict()

        # pass 1: high-confidence
        m, u_track, u_det = _linear_assignment(_iou_distance(pool, dets),
                                               self.args.match_thresh)
        for it, idet in m:
            tr, dt = pool[it], dets[idet]
            if tr.state == "tracked":
                tr.update(dt, self.frame_id); activated.append(tr)
            else:
                tr.re_activate(dt, self.frame_id); refind.append(tr)

        # pass 2: low-confidence, only against still-tracked
        r_tracked = [pool[i] for i in u_track if pool[i].state == "tracked"]
        m2, u_track2, _ = _linear_assignment(_iou_distance(r_tracked, dets_second), 0.5)
        for it, idet in m2:
            tr, dt = r_tracked[it], dets_second[idet]
            if tr.state == "tracked":
                tr.update(dt, self.frame_id); activated.append(tr)
            else:
                tr.re_activate(dt, self.frame_id); refind.append(tr)
        for it in u_track2:
            tr = r_tracked[it]
            if tr.state != "lost":
                tr.mark_lost(); lost.append(tr)

        # unconfirmed tracks vs leftover high-conf dets
        dets_left = [dets[i] for i in u_det]
        mu, u_unc, u_det2 = _linear_assignment(_iou_distance(unconfirmed, dets_left), 0.7)
        for it, idet in mu:
            unconfirmed[it].update(dets_left[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unc:
            unconfirmed[it].mark_removed(); removed.append(unconfirmed[it])

        # spawn new tracks from leftover high-conf dets
        for inew in u_det2:
            tr = dets_left[inew]
            if tr.score < self.args.new_track_thresh:
                continue
            tr.activate(self.frame_id)
            activated.append(tr)

        # age out lost tracks
        for tr in self.lost:
            if self.frame_id - tr.frame_id > self.max_time_lost:
                tr.mark_removed(); removed.append(tr)

        self.tracked = [t for t in self.tracked if t.state == "tracked"]
        self.tracked = self._join(self._join(self.tracked, activated), refind)
        self.lost = self._sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = self._sub(self.lost, self.removed)
        self.removed.extend(removed)
        self.tracked, self.lost = self._remove_dup(self.tracked, self.lost)

        rows = [[*t.tlbr, t.track_id, t.score, t.cls, 0]
                for t in self.tracked if t.is_activated]
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 8), np.float32)


@dataclass
class TrackOutput:
    state: str             # LOCKED / LK / SEARCH / LOST
    bbox: Optional[tuple]
    track_id: int          # stable DISPLAY id (-1 if none)
    driver: str            # DET / LK / SEARCH / LOST
    conf: float
    miss: int = 0          # frames since primary last seen by the detector


class KbnTrackerV3:
    def __init__(self, tracker_type="bytetrack", track_buffer=90,
                 match_thresh=0.8, high=0.50, low=0.25, new_thr=0.50,
                 lk_coast_max=12, search_max=90, reacquire_iou=0.2,
                 coast_drift_boxes=4.0, reacquire_radius_boxes=1.5,
                 reacquire_grow_boxes=0.08, frame_rate=30):
        # Vendored ByteTrack -- no ultralytics/torch. botsort has no GMC/ReID
        # here, so it behaves as plain ByteTrack (kept for CLI compatibility).
        if tracker_type == "botsort":
            args = _make_botsort_args(track_buffer, match_thresh,
                                      high, low, new_thr)
        else:
            args = _make_bytetrack_args(track_buffer, match_thresh,
                                        high, low, new_thr)
        self.tracker = _ByteTracker(args, frame_rate=frame_rate)
        self.tracker_type = tracker_type

        self.primary_id: Optional[int] = None   # byteTrack id we follow
        self.display_id: Optional[int] = None   # stable id shown to user
        self.last_bbox: Optional[tuple] = None
        self.lost_anchor: Optional[tuple] = None # (cx,cy) where target was last seen
        self.miss_count = 0
        self.velocity = (0.0, 0.0)               # last good LK px/frame (for CV coast)

        self.lk = LKMotionMeasurer()
        self.lk_coast_max = lk_coast_max
        self.search_max = max(search_max, lk_coast_max)
        self.reacquire_iou = reacquire_iou
        self.coast_drift_boxes = coast_drift_boxes
        self.reacquire_radius_boxes = reacquire_radius_boxes
        self.reacquire_grow_boxes = reacquire_grow_boxes

    def step(self, prev_gray, curr_gray, detections) -> TrackOutput:
        results = _wrap_results(detections)
        try:
            online = self.tracker.update(results, curr_gray)
        except TypeError:
            online = self.tracker.update(results)
        online = np.asarray(online) if online is not None else np.zeros((0, 8))

        # is our primary still tracked under the same id? -> LOCKED. We do NOT reassign the primary while it is present, so a more confident look-alike can never steal the lock.
        if self.primary_id is not None and len(online):
            for row in online:
                if int(row[4]) == self.primary_id:
                    return self._lock(row, curr_gray, reacquired=False)

        # no primary yet -> acquire the best available track.
        if self.primary_id is None:
            if len(online):
                best = online[int(np.argmax(online[:, 5]))]
                return self._lock(best, curr_gray, reacquired=False)
            return TrackOutput("SEARCH", None, -1, "SEARCH", 0.0, 0)

        # primary missing this frame -> hold the id, coast, and search.
        self.miss_count += 1
        predicted = self._coast_bbox(prev_gray, curr_gray)

        if self.miss_count <= self.search_max:
            # prefer a CONFIRMED ByteTrack track (clean id) -- precise IoU for short gaps, widening centre-distance gate for long gaps.
            cand = self._reacquire(online, predicted, self.miss_count)
            if cand is not None:
                return self._lock(cand, curr_gray, reacquired=True)

            # else snap onto a RAW detector box if one is within the gate. ByteTrack needs 2 consecutive frames to confirm a new track, so a single-frame detector flicker never reaches 3a. Here the lock re-glues to the object the instant the detector sees it and the lose-timer resets -> the lock only drops if detection fails for search_max CONSECUTIVE frames (object truly gone).
            raw = self._reacquire_raw(detections, predicted, self.miss_count)
            if raw is not None:
                bx, conf = raw
                self.last_bbox = tuple(float(v) for v in bx)
                self.lost_anchor = ((self.last_bbox[0] + self.last_bbox[2]) * 0.5, (self.last_bbox[1] + self.last_bbox[3]) * 0.5)
                self.lk.seed(curr_gray, self.last_bbox)
                self.velocity = (0.0, 0.0)
                self.miss_count = 0
                return TrackOutput("LOCKED", self.last_bbox,self.display_id or -1, "DET", float(conf), 0)

            # still missing: keep the id alive and keep coasting on optical flow.
            if self.miss_count <= self.lk_coast_max:
                return TrackOutput("LK", self.last_bbox,self.display_id or -1, "LK", 0.0,self.miss_count)
            return TrackOutput("SEARCH", self.last_bbox,self.display_id or -1, "SEARCH", 0.0,self.miss_count)

        # search window exhausted -> declare lost and free the id so a genuinely new target can be acquired next frame.
        lost_id = self.display_id or -1
        miss = self.miss_count
        self._reset_primary()
        return TrackOutput("LOST", None, lost_id, "LOST", 0.0, miss)

    def _lock(self, row, curr_gray, reacquired) -> TrackOutput:
        bbox = tuple(float(v) for v in row[:4])
        tid = int(row[4])
        conf = float(row[5])
        if self.display_id is None:
            # first acquisition: the displayed id is ByteTrack's id.
            self.display_id = tid
        # if `reacquired` with a different ByteTrack id, display_id is left untouched -> the lock looks continuous to the user even though ByteTrack minted a new internal id after its buffer expired.
        self.primary_id = tid
        self.last_bbox = bbox
        self.lost_anchor = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        self.lk.seed(curr_gray, bbox)
        self.miss_count = 0
        self.velocity = (0.0, 0.0)   # fresh fix -> drop any stale coast velocity
        driver = "DET"
        return TrackOutput("LOCKED", bbox, self.display_id, driver, conf, 0)

    def _coast_bbox(self, prev_gray, curr_gray) -> Optional[tuple]:
        if self.last_bbox is None or prev_gray is None:
            return self.last_bbox
        bx1, by1, bx2, by2 = self.last_bbox
        w, h = bx2 - bx1, by2 - by1   # keep size fixed

        disp = self.lk.measure(prev_gray, curr_gray)
        if disp is not None:
            dx, dy, _scl = disp       # _scl ignored on purpose (see docstring)
            vx, vy = self.velocity
            self.velocity = (0.6 * vx + 0.4 * dx, 0.6 * vy + 0.4 * dy)
        else:
            dx, dy = self.velocity            # flow lost -> constant-velocity coast
            self.velocity = (0.9 * dx, 0.9 * dy)   # decay so it can't run away

        cx = (bx1 + bx2) * 0.5 + dx
        cy = (by1 + by2) * 0.5 + dy

        # bound straight-line runaway. The coast may follow flow/velocity, but the box centre cannot wander more than `coast_drift_boxes` diagonals from where the target was LAST ACTUALLY SEEN. Without this, the per-frame re-seed latches onto drifting background / camera-pan motion, feeds it into `velocity`, and the box flies off in a straight line and never re-acquires. Hitting the cap kills the dead-reckoning.
        if self.lost_anchor is not None:
            ax, ay = self.lost_anchor
            diag = (w * w + h * h) ** 0.5
            cap = diag * self.coast_drift_boxes
            ddx, ddy = cx - ax, cy - ay
            d = (ddx * ddx + ddy * ddy) ** 0.5
            if d > cap and d > _EPS:
                s = cap / d
                cx, cy = ax + ddx * s, ay + ddy * s
                self.velocity = (0.0, 0.0)

        self.last_bbox = (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)

        # re-engage flow from the new location for next frame.
        self.lk.seed(curr_gray, self.last_bbox)
        return self.last_bbox

    def _reacquire(self, online, predicted, miss):
        if predicted is None or len(online) == 0:
            return None
        px1, py1, px2, py2 = predicted
        pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        diag = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5 + _EPS
        radius = diag * (self.reacquire_radius_boxes + self.reacquire_grow_boxes * miss)
        best_row, best_dist = None, None
        for row in online:
            bx = tuple(float(v) for v in row[:4])
            ccx, ccy = (bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5
            dist = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
            if _iou(predicted, bx) >= self.reacquire_iou or dist <= radius:
                if best_dist is None or dist < best_dist:
                    best_dist, best_row = dist, row
        return best_row

    def _reacquire_raw(self, detections, predicted, miss):
        if predicted is None or not detections:
            return None
        px1, py1, px2, py2 = predicted
        pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        diag = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5 + _EPS
        radius = diag * (self.reacquire_radius_boxes + self.reacquire_grow_boxes * miss)
        best, best_dist = None, None
        for bx, conf in detections:
            ccx, ccy = (bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5
            dist = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
            if _iou(predicted, bx) >= self.reacquire_iou or dist <= radius:
                if best_dist is None or dist < best_dist:
                    best_dist, best = dist, (bx, conf)
        return best

    def _reset_primary(self):
        self.primary_id = None
        self.display_id = None
        self.last_bbox = None
        self.lost_anchor = None
        self.miss_count = 0
        self.velocity = (0.0, 0.0)


def open_camera(index, width, height):
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_fov(path):
    f = np.load(path)
    return (f["map_x"], f["map_y"], float(f["alpha"]),float(f["beta"]), int(f["imgsz"]))


def _label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y), C_TEXT_BG, -1)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y), color, 1)
    cv2.putText(frame, text, (x + 2, y - 3),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hef", default="best.hef")
    p.add_argument("--fov", default="telescope_foveation.npz",help="foveation .npz -- pass '' for a plain (non-telescope) HEF")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf", type=float, default=CONF_FLOOR, help="detector score floor (0.25 -- feeds the low-score pass to MAINTAIN a lock through faint frames)")
    p.add_argument("--iou", type=float, default=IOU_NMS, help="NMS IoU")
    # Tracker selection + tuning
    p.add_argument("--tracker", choices=("bytetrack", "botsort"),default="bytetrack")
    p.add_argument("--track-high", type=float, default=0.50,help="high-conf split: >=this takes pass-1 association")
    p.add_argument("--track-low", type=float, default=0.25,help="low-conf floor for pass-2: 0.25-0.50 boxes can only CONTINUE an existing lock, not start one")
    p.add_argument("--new-thr", type=float, default=0.50,help="min conf to spawn a new track (keeps weak boxes from creating phantom ids)")
    p.add_argument("--track-buffer", type=int, default=90,help="ByteTrack frames to keep lost tracks alive for same-id re-association (90 from test1/test2 tuning)")
    p.add_argument("--match-thresh", type=float, default=0.8,help="IoU threshold for det<->track matching inside ByteTrack")
    p.add_argument("--lk-coast-max", type=int, default=12,help="frames coasted as LK (orange) before entering SEARCH")
    p.add_argument("--search-max", type=int, default=90,help="total frames to hold the id and search before declaring LOST (90 from test1/test2 tuning)")
    p.add_argument("--reacquire-iou", type=float, default=0.2,help="min IoU between LK-predicted box and a current track to re-acquire the SAME id")
    p.add_argument("--coast-drift-boxes", type=float, default=4.0,help="max box-diagonals the search box may drift from where the target was last seen (caps straight-line runaway)")
    p.add_argument("--reacquire-radius-boxes", type=float, default=1.5,help="base re-acquire radius in box-diagonals (short-gap gate)")
    p.add_argument("--reacquire-grow-boxes", type=float, default=0.08,help="re-acquire radius growth per missed frame, in box-diagonals (long-gap gate)")
    p.add_argument("--frame-rate", type=int, default=30,help="approx FPS -- affects ByteTrack's KF noise scaling")
    # Output
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--show-raw", action="store_true",help="draw thin cyan boxes for ALL detector outputs (debug)")
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
        print("foveation: disabled")

    print(f"loading HEF: {args.hef}")
    model = HailoYOLOv8(args.hef)
    tracker = KbnTrackerV3(
        tracker_type=args.tracker,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        high=args.track_high,
        low=args.track_low,
        new_thr=args.new_thr,
        lk_coast_max=args.lk_coast_max,
        search_max=args.search_max,
        reacquire_iou=args.reacquire_iou,
        coast_drift_boxes=args.coast_drift_boxes,
        reacquire_radius_boxes=args.reacquire_radius_boxes,
        reacquire_grow_boxes=args.reacquire_grow_boxes,
        frame_rate=args.frame_rate,
    )
    print(f"tracker: {args.tracker} "
          f"(high={args.track_high} low={args.track_low} buf={args.track_buffer} "
          f"lk={args.lk_coast_max} search={args.search_max} reiou={args.reacquire_iou})")

    cap = open_camera(args.camera, args.width, args.height)
    print("Press 'q' to quit." if not args.no_show
          else "Headless. Ctrl+C to stop.")

    writer, out_path = None, None
    if not args.no_record:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"track_iter3_{stamp}.mp4"

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
                writer = cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*"mp4v"),args.fps, (W, H))
                if not writer.isOpened():
                    print(f"could not open VideoWriter {out_path} -- recording off")
                    writer, out_path = None, None
                else:
                    print(f"Recording to: {out_path}")

            detections, infer_ms = run_detector(
                model, frame, fov, args.conf, args.iou)
            infer_ms_s = infer_ms if n == 0 else 0.9 * infer_ms_s + 0.1 * infer_ms

            out = tracker.step(prev_gray, gray, detections)

            # optional: draw raw detector outputs (only above 0.25 for clarity).
            if args.show_raw:
                for dbox, dconf in detections:
                    if dconf < 0.25:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in dbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), C_RAW, 1)

            # primary box -- drawn whenever we have one (incl. SEARCH coast), so the operator can see the search region following the target.
            if out.bbox is not None:
                x1, y1, x2, y2 = (int(v) for v in out.bbox)
                color = {"DET": C_DET, "LK": C_LK,
                         "LOST": C_LOST, "SEARCH": C_SEARCH}.get(out.driver, C_DET)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"id={out.track_id}  {out.driver}"
                if out.driver == "DET":
                    tag += f"  {out.conf:.2f}"
                elif out.miss:
                    tag += f"  miss={out.miss}/{args.search_max}"
                _label(frame, tag, x1, max(14, y1 - 2), color)

            # HUD
            infer_fps = 1000.0 / max(infer_ms_s, 1e-6)
            hud_color = C_DET if out.state == "LOCKED" else (C_SEARCH if out.state in ("SEARCH", "LK") else C_LOST)
            _label(frame,
                   f"{out.state}  |  YOLO "
                   f"{infer_fps:5.1f} FPS ({infer_ms_s:4.1f}ms)  |  "
                   f"{len(detections)} det",
                   8, 22, hud_color)

            if writer is not None:
                writer.write(frame)
            if args.no_show:
                print(f"f{n:05d} {out.state:<7} id={out.track_id:<3} "
                      f"drv={out.driver:<6} miss={out.miss:<2} "
                      f"YOLO {infer_fps:5.1f}fps {len(detections)} det")
            else:
                cv2.imshow("kbn_track_iter3: sticky-ID + IoU re-acquire", frame)
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
