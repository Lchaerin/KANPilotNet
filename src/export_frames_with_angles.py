#!/usr/bin/env python3
# export_images_and_data.py

import os, sys, glob, cv2, numpy as np

def _try_load_raw(path, kind):
    cands = [("<f8",8), (">f8",8), ("<f4",4), (">f4",4)]
    nbytes = os.path.getsize(path)
    best, best_score, best_arr = None, None, None
    for dt, sz in cands:
        if nbytes % sz: continue
        try:
            arr = np.memmap(path, dtype=dt, mode="r")
            a = np.asarray(arr, dtype=np.float64, copy=False)
            if a.size == 0: continue
            finite = np.isfinite(a)
            bad_nan = 1.0 - float(np.mean(finite))
            if kind == "value":
                bad_range = float(np.mean(np.abs(a[finite]) > 1000.0)) if finite.any() else 1.0
                score = bad_nan + bad_range
            else:
                if a.size >= 2:
                    d = np.diff(a)
                    bad_monotonic = float(np.mean(d[~np.isnan(d)] < 0))
                else:
                    bad_monotonic = 0.0
                score = bad_nan + bad_monotonic
        except Exception:
            continue
        if (best_score is None) or (score < best_score):
            best_score, best, best_arr = score, dt, arr
    if best_arr is None:
        raise RuntimeError("raw detection failed: %s" % path)
    return np.asarray(best_arr).reshape(-1), "raw:%s" % best

def _load_any_numeric(file_path, prefer_kind):
    with open(file_path, "rb") as f:
        head6 = f.read(6)
        f.seek(0)
        head4 = f.read(4)
    if head6 == b"\x93NUMPY":
        arr = np.load(file_path, mmap_mode="r", allow_pickle=False)
        return np.asarray(arr).reshape(-1), "npy"
    if head4 == b"PK\x03\x04":
        z = np.load(file_path, allow_pickle=False)
        key = "arr_0" if "arr_0" in z else list(z.keys())[0]
        arr = z[key]
        return np.asarray(arr).reshape(-1), "npz:%s" % key
    return _try_load_raw(file_path, prefer_kind)

def load_value_and_time(steer_dir):
    vpath_npy = os.path.join(steer_dir, "value.npy")
    vpath_raw = os.path.join(steer_dir, "value")
    tpath_npy = os.path.join(steer_dir, "t.npy")
    tpath_raw = os.path.join(steer_dir, "t")

    if os.path.isfile(vpath_npy):
        v = np.load(vpath_npy, mmap_mode="r", allow_pickle=False).reshape(-1)
    elif os.path.isfile(vpath_raw):
        v, _ = _load_any_numeric(vpath_raw, "value")
    else:
        raise RuntimeError("No value(.npy): %s" % steer_dir)

    t = None
    if os.path.isfile(tpath_npy):
        t = np.load(tpath_npy, mmap_mode="r", allow_pickle=False).reshape(-1)
    elif os.path.isfile(tpath_raw):
        t, _ = _load_any_numeric(tpath_raw, "t")

    return np.asarray(v, dtype=np.float64), (None if t is None else np.asarray(t, dtype=np.float64))

def find_segments(root):
    vids = glob.glob(os.path.join(root, "**", "video.hevc"), recursive=True)
    vids.sort()
    out = []
    for v in vids:
        seg_dir = os.path.dirname(v)
        steer_dir = os.path.join(seg_dir, "processed_log", "CAN", "steering_angle")
        if os.path.isdir(steer_dir):
            out.append((v, steer_dir, seg_dir))
    return out

def nearest_value_at_time(values, t, ts, j_hint=0):
    j = j_hint
    n = len(t)
    if n == 0: return None, j
    if j >= n: j = n-1
    while j + 1 < n and t[j + 1] <= ts:
        j += 1
    if j + 1 < n:
        jstar = j if (ts - t[j]) <= (t[j+1] - ts) else (j + 1)
    else:
        jstar = j
    if jstar >= len(values):
        jstar = len(values) - 1
    return values[jstar], j

def export_all(data_root, out_dir, width=455, height=256, jpeg_quality=70,
               index_align=False, time_offset=0.0, value_is_radians=False):
    data_root = os.path.abspath(os.path.expanduser(data_root))
    out_dir   = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    data_fp = open(os.path.join(out_dir, "data.txt"), "w")
    counter = 1

    segs = find_segments(data_root)
    print("segments:", len(segs))

    for vpath, steer_dir, seg_dir in segs:
        values, t = load_value_and_time(steer_dir)

        if value_is_radians:
            values = values * (180.0/np.pi)  # deg

        if (not index_align) and (t is not None) and len(t) > 0 and np.isfinite(t[0]):
            t = t - t[0]

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print("warn: cannot open", vpath); continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = None

        idx = 0
        used = 0
        j_hint = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if index_align or (t is None):
                if idx >= len(values): break
                val = values[idx]
            else:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if np.isfinite(pos_ms) and pos_ms > 0:
                    ts = pos_ms * 1e-3 + time_offset
                elif fps:
                    ts = (idx / float(fps)) + time_offset
                else:
                    if idx >= len(values): break
                    val = values[idx]
                    ts = None
                if ts is not None:
                    val, j_hint = nearest_value_at_time(values, t, ts, j_hint)

            frame = cv2.resize(frame, (int(width), int(height)))
            name = f"{counter}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            data_fp.write(f"{name} {float(val):.6f}\n")

            counter += 1
            idx += 1
            used += 1

        cap.release()
        print(f"{os.path.basename(seg_dir)}: frames_used={used}, values_avail={len(values)}, mode={'index' if (index_align or t is None) else 'time'}")

    data_fp.close()
    print("DONE. total images:", counter-1)
    print("data.txt ->", os.path.join(out_dir, "data.txt"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="video.hevc → <OUT>/1.jpg…, data.txt(<file> <deg>)")
    ap.add_argument("data_root")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=455)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--index-align", action="store_true")
    ap.add_argument("--time-offset", type=float, default=0.0)
    ap.add_argument("--value-is-radians", action="store_true")
    args = ap.parse_args()
    export_all(args.data_root, args.out_dir, args.width, args.height, args.jpeg_quality,
               args.index_align, args.time_offset, args.value_is_radians)
