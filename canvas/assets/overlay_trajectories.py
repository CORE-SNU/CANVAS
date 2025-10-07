
import os
import cv2
import glob
import zipfile
import argparse
import numpy as np
from typing import Dict, Tuple, List

# We will reuse to_image_frame from the provided visualization_utils.py for consistent homography math.
# If the import fails due to path differences, we fall back to a local implementation that mirrors it.
def _fallback_to_image_frame(pos: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param pos: numpy array of shape (N, 2) containing the world coordinates of N points
    :param H: (3x3)-homography matrix that transforms image coordinates to world coordinates
    :return: two arrays of shape (N,), image x and y
    """
    N = pos.shape[0]
    pos_h = np.hstack([pos, np.ones((N, 1))])  # homogeneous [x, y, 1]
    # Solve H * img_h = world_h  => img_h = H^{-1} * world_h
    img_h = np.linalg.solve(H, pos_h.T)  # (3, N)
    x = img_h[0] / img_h[2]
    y = img_h[1] / img_h[2]
    return x, y

try:
    # try local directory import first
    import sys
    sys.path.append(os.path.dirname(__file__))
    import visualization_utils as vis
    to_image_frame = vis.to_image_frame
except Exception as e:
    # fallback if vis import not available in the same folder
    to_image_frame = _fallback_to_image_frame


DATASET_TO_NPY = {
    "ETH": "biwi_eth.npy",
    "Hotel": "biwi_hotel.npy",
    "Zara01": "crowds_zara01.npy",
    "Zara02": "crowds_zara02.npy",
    "Univ": "students003.npy",
}

# The homographies.zip uses lowercase names and zara1/zara2.
DATASET_TO_H_FILE = {
    "ETH": "eth.txt",
    "Hotel": "hotel.txt",
    "Zara01": "zara1.txt",
    "Zara02": "zara2.txt",
    "Univ": "univ.txt",
}

# Frame folder name heuristics (created by video_parser.py at 2.5 FPS)
DATASET_FRAME_ALIASES = {
    "ETH": ["ETH", "eth", "biwi_eth"],
    "Hotel": ["Hotel", "hotel", "biwi_hotel"],
    "Zara01": ["Zara01", "zara01", "zara1", "crowds_zara01"],
    "Zara02": ["Zara02", "zara02", "zara2", "crowds_zara02"],
    "Univ": ["Univ", "univ", "students003"],
}

def stable_color_bgr(idx: int) -> Tuple[int, int, int]:
    """
    Deterministic pseudo-random color from an integer id.
    Returns BGR tuple for OpenCV.
    """
    # Simple xorshift or hash to seed color
    rng = np.random.RandomState(seed=(idx * 9781) % 2**32)
    c = rng.randint(0, 255, size=3).tolist()
    return int(c[2]), int(c[1]), int(c[0])  # convert RGB -> BGR

def find_frames_dir(asset_dir: str, dataset: str) -> str:
    frames_root = os.path.join(asset_dir, "frames")
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frames root not found: {frames_root}. Run video_parser.py first.")
    aliases = DATASET_FRAME_ALIASES.get(dataset, [dataset, dataset.lower()])
    # search for a subdirectory that matches any alias (substring ok)
    candidates = []
    for d in os.listdir(frames_root):
        path = os.path.join(frames_root, d)
        if not os.path.isdir(path):
            continue
        for a in aliases:
            if a.lower() in d.lower():
                candidates.append(path)
                break
    if not candidates:
        # fallback: if exactly one directory exists, use it
        subdirs = [os.path.join(frames_root, x) for x in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, x))]
        if len(subdirs) == 1:
            return subdirs[0]
        raise FileNotFoundError(f"No frames directory matching {dataset} under {frames_root}. Found: {os.listdir(frames_root)}")
    # Prefer the shortest name (often exact match like "eth")
    candidates.sort(key=lambda p: len(os.path.basename(p)))
    return candidates[0]

def load_homography(h_dir: str, dataset: str) -> np.ndarray:
    fname = DATASET_TO_H_FILE[dataset]
    h_path = os.path.join(h_dir, fname)
    if not os.path.isfile(h_path):
        raise FileNotFoundError(f"Homography file not found for {dataset}: {h_path}")
    H = np.loadtxt(h_path, dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"Homography for {dataset} is not 3x3: {H.shape}")
    return H

def ensure_homography_dir(homography_zip: str, homography_dir: str) -> str:
    """
    If a zip is provided, extract it and return the homography directory path.
    If a directory is provided, just return it.
    """
    if homography_dir and os.path.isdir(homography_dir):
        return homography_dir
    if homography_zip and os.path.isfile(homography_zip):
        extract_root = os.path.join(os.path.dirname(homography_zip), "_homographies_extracted")
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(homography_zip) as zf:
            zf.extractall(extract_root)
        # The zip contains 'homographies/' subdir
        h_dir = os.path.join(extract_root, "homographies")
        if not os.path.isdir(h_dir):
            # Maybe it extracted directly
            h_dir = extract_root
        return h_dir
    raise FileNotFoundError("Provide either --homography_dir or --homography_zip")

def load_dataset_npy(npy_dir: str, dataset: str) -> np.ndarray:
    fname = DATASET_TO_NPY[dataset]
    path = os.path.join(npy_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset npy for {dataset} not found at {path}")
    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Unexpected npy shape for {dataset}: {arr.shape}. Expected (T, N, 2).")
    return arr

def draw_polyline(img, pts_xy: np.ndarray, color_bgr: Tuple[int,int,int], thickness: int = 2):
    """
    pts_xy: (K, 2) in image coordinates
    """
    if pts_xy.shape[0] < 2:
        return
    # OpenCV expects int32 pixel coordinates
    pts = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color_bgr, thickness=thickness)

def draw_agent_point(img, pt_xy: Tuple[float,float], color_bgr: Tuple[int,int,int], radius: int = 3):
    x, y = int(round(pt_xy[0])), int(round(pt_xy[1]))
    cv2.circle(img, (x, y), radius, color_bgr, -1)

def put_agent_id(img, pt_xy: Tuple[float,float], agent_id: int, color_bgr: Tuple[int,int,int]):
    x, y = int(round(pt_xy[0])), int(round(pt_xy[1]))
    cv2.putText(img, str(agent_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)

def overlay_dataset_on_frames(
    asset_dir: str,
    npy_dir: str,
    homography_dir: str,
    dataset: str,
    history: int = 12,
    frame_offset: int = 0,
    output_video: bool = True,
    video_fps: float = 25.0,
) -> str:
    """
    For each frame index t, reads the corresponding frame image extracted by video_parser.py
    (saved at asset_dir/frames/<video_name>/<t>.png), projects ground-truth trajectories (up to 'history' steps),
    and writes overlay images to asset_dir/overlays/<dataset>/<t>.png and an .mp4 video.
    Returns the path to the output video (if created), else the folder path of overlays.
    """
    # Paths
    frames_dir = find_frames_dir(asset_dir, dataset)
    out_dir = os.path.join(asset_dir, "overlays", dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    H = load_homography(homography_dir, dataset)
    traj = load_dataset_npy(npy_dir, dataset)  # shape (T, N, 2), NaN for missing
    T, N, _ = traj.shape

    # Optional: VideoWriter
    out_video_path = os.path.join(asset_dir, f"{dataset}_overlay.mp4")
    writer = None
    if output_video:
        # Determine frame size from first available frame
        # Try t=0; if not exists, search next few frames
        frame0 = None
        t_probe = 0
        for k in range(50):
            fpath = os.path.join(frames_dir, f"{t_probe + k + frame_offset}.png")
            if os.path.isfile(fpath):
                frame0 = cv2.imread(fpath)
                if frame0 is not None:
                    break
        if frame0 is None:
            print(f"[{dataset}] No frame images found in {frames_dir}. Skipping video writer.")
        else:
            h, w = frame0.shape[:2]
            writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (w, h))

    # Iterate frames
    # We'll overlay until the min of (available image frames) or T (trajectory frames).
    # We don't know how many PNGs exist; count them.
    pngs = glob.glob(os.path.join(frames_dir, "*.png"))
    available_indices = set()
    for p in pngs:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            idx = int(base)
            available_indices.add(idx)
        except ValueError:
            # Non-integer filenames are ignored
            continue

    # If no extracted frames found, we still attempt to read sequentially by index until failure.
    max_idx = max(available_indices) if available_indices else min(T-1, 999999)

    for t in range(0, min(T, max_idx + 1)):
        fidx = t + frame_offset
        frame_path = os.path.join(frames_dir, f"{fidx}.png")
        if not os.path.isfile(frame_path):
            # No image for this time step; skip
            continue

        img = cv2.imread(frame_path)
        if img is None:
            continue

        # For each agent, collect finite history up to 't'
        # We'll draw short polylines for recent positions.
        for agent_id in range(N):
            # collect indices [max(0, t-history), t] where both coords are finite
            t0 = max(0, t - history)
            segment = traj[t0:t+1, agent_id, :]  # (K, 2)
            mask = np.isfinite(segment).all(axis=1)
            if not np.any(mask):
                continue
            world_pts = segment[mask]  # (K, 2)

            # project to image
            xs, ys = to_image_frame(world_pts, H)
            img_pts = np.stack([xs, ys], axis=1)

            color = stable_color_bgr(agent_id)

            # draw polyline
            draw_polyline(img, img_pts, color_bgr=color, thickness=2)

            # draw current point and id
            last_pt = img_pts[-1]
            draw_agent_point(img, (last_pt[0], last_pt[1]), color_bgr=color, radius=3)
            put_agent_id(img, (last_pt[0], last_pt[1]), agent_id=agent_id, color_bgr=color)

        # write overlay image
        out_png = os.path.join(out_dir, f"{t:05d}.png")
        cv2.imwrite(out_png, img)

        # append to video
        if writer is not None:
            writer.write(img)

        if t % 50 == 0:
            print(f"[{dataset}] rendered t={t} -> {out_png}")

    if writer is not None:
        writer.release()
        print(f"[{dataset}] wrote {out_video_path}")
        return out_video_path
    else:
        return out_dir


def main():
    parser = argparse.ArgumentParser(description="Overlay ETH/UCY ground-truth trajectories onto frames using homographies.")
    parser.add_argument("asset_dir", type=str, help="Root asset dir (same one used by video_parser.py).")
    parser.add_argument("--npy_dir", type=str, default=".", help="Directory containing the dataset .npy files.")
    parser.add_argument("--homography_zip", type=str, default=None, help="Path to homographies.zip.")
    parser.add_argument("--homography_dir", type=str, default=None, help="Path to a folder with homography .txt files.")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["ETH", "Hotel", "Zara01", "Zara02", "Univ"],
                        help="Subset of datasets to render.")
    parser.add_argument("--history", type=int, default=12, help="Number of past steps to draw for each agent.")
    parser.add_argument("--frame_offset", type=int, default=0, help="Shift PNG frame index by this amount to account for alignment differences.")
    parser.add_argument("--no_video", action="store_true", help="Do not create an .mp4, only PNG overlays.")
    parser.add_argument("--video_fps", type=float, default=25.0, help="FPS of the output .mp4 (if created).")

    args = parser.parse_args()

    # Resolve homography dir from inputs
    h_dir = ensure_homography_dir(args.homography_zip, args.homography_dir)

    # Process each dataset
    for ds in args.datasets:
        try:
            output = overlay_dataset_on_frames(
                asset_dir=args.asset_dir,
                npy_dir=args.npy_dir,
                homography_dir=h_dir,
                dataset=ds,
                history=args.history,
                frame_offset=args.frame_offset,
                output_video=not args.no_video,
                video_fps=args.video_fps,
            )
            print(f"[{ds}] Done. Output: {output}")
        except Exception as e:
            print(f"[{ds}] ERROR: {e}")


if __name__ == "__main__":
    main()
