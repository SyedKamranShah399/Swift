from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Sheikh-Javed-Kanade trajectory-subspace bg subtraction")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="annotated mp4 (default <stem>_trajbg.mp4)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--display", action="store_true")
    p.add_argument("--window", type=int, default=10, help="F: number of frames per trajectory window")
    p.add_argument("--step", type=int, default=1, help="advance the window by this many frames")
    p.add_argument("--proc-width", type=int, default=640, help="resize width before processing")
    p.add_argument("--crop", type=float, default=1.0, help="keep this centre fraction of each frame")
    p.add_argument("--max-corners", type=int, default=2000, help="points to seed per window")
    p.add_argument("--quality", type=float, default=0.01)
    p.add_argument("--min-distance", type=int, default=7)
    p.add_argument("--rank", type=int, default=3, help="background subspace dimension (3 per the paper)")
    p.add_argument("--subspace-thresh", type=float, default=2.0,
                   help="per-frame trajectory residual (px) to count as BACKGROUND; "
                        "above = foreground. raise to suppress leak, lower to catch subtle movers")
    p.add_argument("--ransac-iters", type=int, default=400)
    p.add_argument("--min-trajectories", type=int, default=40, help="skip window if fewer complete tracks")
    p.add_argument("--splat-radius", type=int, default=9, help="foreground point splat radius (px)")
    p.add_argument("--min-density", type=int, default=2,
                   help="require this many foreground points to OVERLAP in a region; isolated "
                        "tracking-noise points (density 1) are dropped. raise to kill small boxes")
    p.add_argument("--min-area", type=int, default=60, help="min foreground blob area (px)")
    p.add_argument("--max-area", type=int, default=0, help="reject blobs bigger than this; 0=off")
    p.add_argument("--ch-gain", type=float, default=2.0, help="foreground-residual -> channel gain")
    p.add_argument("--save-channels", default=None,
                   help="dir to dump per-frame 3-channel png [gray, fg-residual, fg-density]")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()

def track_window(frames, max_corners, quality, min_distance):
    F = len(frames)
    p0 = cv2.goodFeaturesToTrack(frames[0], maxCorners=max_corners,
                                 qualityLevel=quality, minDistance=min_distance)
    if p0 is None or len(p0) < 4:
        return None, None
    pts = p0.reshape(-1, 2)
    track = [pts.copy()]
    alive = np.ones(len(pts), bool)
    cur = p0
    for f in range(1, F):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(frames[f - 1], frames[f], cur, None)
        st = st.reshape(-1).astype(bool)
        alive &= st
        track.append(nxt.reshape(-1, 2))
        cur = nxt
    if alive.sum() < 4:
        return None, None
    track = [t[alive] for t in track]                  
    P = track[0].shape[0]
    W = np.empty((2 * F, P), np.float64)
    for f in range(F):
        W[f] = track[f][:, 0]                         
        W[F + f] = track[f][:, 1]                      
    return W, track[-1]

def background_residual(W, rank, thresh_px, iters):
    F = W.shape[0] // 2
    Wc = W - W.mean(axis=1, keepdims=True)          
    twoF, P = Wc.shape
    sqrtF = np.sqrt(F)
    rng = np.random.default_rng(0)

    def per_frame_residual(Q):
        proj = Q @ (Q.T @ Wc)
        return np.linalg.norm(Wc - proj, axis=0) / sqrtF 

    best_inliers, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(P, size=rank, replace=False)
        B = Wc[:, idx]
        U, s, _ = np.linalg.svd(B, full_matrices=False)
        if s[-1] < 1e-6:
            continue                            
        Q = U[:, :rank]
        inliers = per_frame_residual(Q) < thresh_px
        if inliers.sum() > best_count:
            best_count, best_inliers = inliers.sum(), inliers
    if best_inliers is None or best_count < rank:
        return np.zeros(P)
    U, _, _ = np.linalg.svd(Wc[:, best_inliers], full_matrices=False) 
    return per_frame_residual(U[:, :rank])

def foreground_maps(cur_pts, residual, fg, shape, radius):
    h, w = shape
    res_map = np.zeros(shape, np.float32)
    den_map = np.zeros(shape, np.float32)
    r2 = radius * radius
    for (x, y), rv in zip(cur_pts[fg], residual[fg]):
        x, y = int(x), int(y)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        disk = (xx * xx + yy * yy) <= r2
        den_map[y0:y1, x0:x1] += disk
        patch = res_map[y0:y1, x0:x1]
        np.maximum(patch, disk * float(rv), out=patch)
    return res_map, den_map


def boxes_from(mask, min_area, max_area):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or (max_area and a > max_area):
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes

def prep_frame(frame, crop, proc_width):
    if 0.0 < crop < 1.0:
        H0, W0 = frame.shape[:2]
        cw, ch = int(W0 * crop), int(H0 * crop)
        x0, y0 = (W0 - cw) // 2, (H0 - ch) // 2
        frame = frame[y0:y0 + ch, x0:x0 + cw]
    if proc_width and frame.shape[1] > proc_width:
        s = proc_width / frame.shape[1]
        frame = cv2.resize(frame, (proc_width, int(frame.shape[0] * s)))
    return frame

def main():
    args = parse_args()
    vpath = Path(args.video)
    if not vpath.exists():
        raise FileNotFoundError(vpath)

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {vpath}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = args.out or f"{vpath.stem}_trajbg.mp4"
    if args.save_channels:
        Path(args.save_channels).mkdir(parents=True, exist_ok=True)

    print(f"video: {vpath.name}  window F={args.window}  rank={args.rank}  "
          f"thresh={args.subspace_thresh}px")

    F = max(4, args.window)
    gray_buf = deque(maxlen=F)
    color_buf = deque(maxlen=F)
    writer = None
    read = proc = det_frames = total_movers = saved = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = prep_frame(frame, args.crop, args.proc_width)
        gray_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        color_buf.append(frame)
        read += 1

        if len(gray_buf) == F and (read % args.step == 0):
            W, cur_pts = track_window(list(gray_buf), args.max_corners,
                                      args.quality, args.min_distance)
            boxes, res_map, fg_count, residual = [], None, 0, None
            if W is not None and W.shape[1] >= args.min_trajectories:
                residual = background_residual(W, args.rank, args.subspace_thresh,
                                               args.ransac_iters)
                fg = residual >= args.subspace_thresh          
                fg_count = int(fg.sum())
                h, w = gray_buf[-1].shape
                res_map, den_map = foreground_maps(cur_pts, residual, fg, (h, w),
                                                   args.splat_radius)
                mask = (den_map >= args.min_density).astype(np.uint8) * 255  
                boxes = boxes_from(mask, args.min_area, args.max_area)
            cur_g = gray_buf[-1]
            if res_map is None:
                res_map = np.zeros_like(cur_g, np.float32)
            ch1 = np.clip(args.ch_gain * res_map, 0, 255).astype(np.uint8)
            if args.save_channels:
                ch2 = np.clip(255.0 * (res_map > 0), 0, 255).astype(np.uint8)
                cv2.imwrite(f"{args.save_channels}/chan_{saved:06d}.png",
                            np.dstack([cur_g, ch1, ch2]))
                saved += 1

            vis = color_buf[-1].copy()
            if residual is not None:
                for (x, y), is_fg in zip(cur_pts, residual >= args.subspace_thresh):
                    c = (0, 0, 255) if is_fg else (0, 180, 0)
                    cv2.circle(vis, (int(x), int(y)), 2, c, -1)
            for (x, y, bw, bh) in boxes:
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(vis, f"fg pts:{fg_count}  movers:{len(boxes)}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            heat = cv2.applyColorMap(ch1, cv2.COLORMAP_JET)
            cv2.putText(heat, "Ch1 traj bg-sub", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            view = np.hstack([vis, heat])

            if not args.no_video:
                if writer is None:
                    hh, ww = view.shape[:2]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (ww, hh))
                writer.write(view)
            if args.display:
                cv2.imshow("trajectory bg-sub -- frame | Ch1", view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            det_frames += 1
            total_movers += len(boxes)
            proc += 1
            if proc % 20 == 0:
                fps = proc / (time.perf_counter() - t0)
                print(f"  {proc} windows  {fps:5.1f} win/s  "
                      f"avg movers/frame={total_movers / max(det_frames, 1):.2f}")
            if args.max_frames and proc >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    wall = time.perf_counter() - t0
    print(f"\ndone: {proc} windows in {wall:.1f}s ({proc / max(wall, 1e-6):.1f} win/s)")
    if not args.no_video:
        print(f"saved: {out_path}")
    if args.save_channels:
        print(f"saved {saved} 3-channel tensors to {args.save_channels}/")


if __name__ == "__main__":
    main()
