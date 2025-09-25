
import os
import cv2
import glob
import zipfile
import numpy as np
from typing import Dict, Tuple, List, Optional

def _fallback_to_image_frame(pos: np.ndarray, H: np.ndarray):
    N = pos.shape[0]
    pos_h = np.hstack([pos, np.ones((N, 1))])  # [x, y, 1]
    img_h = np.linalg.solve(H, pos_h.T)  # H * img_h = world_h  => img_h = H^{-1} * world_h
    x = img_h[0] / img_h[2]
    y = img_h[1] / img_h[2]
    return x, y

try:
    import sys
    sys.path.append(os.path.dirname(__file__))
    import visualization_utils as vis
    to_image_frame = vis.to_image_frame
except Exception:
    to_image_frame = _fallback_to_image_frame

DATASET_FRAME_ALIASES = {
    "ETH": ["ETH", "eth", "biwi_eth"],
    "Hotel": ["Hotel", "hotel", "biwi_hotel"],
    "Zara01": ["Zara01", "zara01", "zara1", "crowds_zara01"],
    "Zara02": ["Zara02", "zara2", "zara02", "crowds_zara02"],
    "Univ": ["Univ", "univ", "students003"],
}

DATASET_TO_H_FILE = {
    "ETH": "eth.txt",
    "Hotel": "hotel.txt",
    "Zara01": "zara1.txt",
    "Zara02": "zara2.txt",
    "Univ": "univ.txt",
}

FRAMES_ROOT = "/home/snowhan/ECP-MPC/assets/final/frames"
HZIP_PATH = "/home/snowhan/ECP-MPC/assets/homographies"

def find_frames_dir(asset_dir: str, dataset: str) -> str:
    #frames_root = os.path.join(asset_dir, "frames")
    frames_root = FRAMES_ROOT
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frames root not found: {frames_root}. Run video_parser.py first.")
    aliases = DATASET_FRAME_ALIASES.get(dataset, [dataset, dataset.lower()])
    candidates = []
    for d in os.listdir(frames_root):
        p = os.path.join(frames_root, d)
        if not os.path.isdir(p):
            continue
        for a in aliases:
            if a.lower() in d.lower():
                candidates.append(p)
                break
    if not candidates:
        subs = [os.path.join(frames_root, x) for x in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, x))]
        if len(subs) == 1:
            return subs[0]
        raise FileNotFoundError(f"No frames directory matching {dataset} under {frames_root}. Found: {os.listdir(frames_root)}")
    candidates.sort(key=lambda p: len(os.path.basename(p)))
    return candidates[0]

def ensure_homography_dir(homography_zip: Optional[str], homography_dir: Optional[str]) -> str:
    if homography_dir and os.path.isdir(homography_dir):
        homography_dir=HZIP_PATH
        #return homography_dir
    if homography_zip and os.path.isfile(homography_zip):
        extract_root = os.path.join(os.path.dirname(homography_zip), "_homographies_extracted")
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(homography_zip) as zf:
            zf.extractall(extract_root)
        h_dir = os.path.join(extract_root, "homographies")
        if not os.path.isdir(h_dir):
            h_dir = extract_root
        return h_dir
    raise FileNotFoundError("Provide either homography_dir or homography_zip")

def load_homography(h_dir: str, dataset: str) -> np.ndarray:
    f = os.path.join(h_dir, DATASET_TO_H_FILE[dataset])
    H = np.loadtxt(f, dtype=float)
    if H.shape != (3,3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H

def stable_color_bgr(idx: int) -> Tuple[int,int,int]:
    rng = np.random.RandomState(seed=(idx * 9781) % 2**32)
    c = rng.randint(0, 255, size=3).tolist()
    return int(c[2]), int(c[1]), int(c[0])  # RGB->BGR

def _polyline(img, pts_xy: np.ndarray, color: Tuple[int,int,int], thickness=2):
    if pts_xy is None or len(pts_xy) < 2:
        return
    pts = np.round(pts_xy).astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

def _dots(img, pts_xy: np.ndarray, color: Tuple[int,int,int], radius=2, step: int = 1):
    if pts_xy is None or len(pts_xy) == 0:
        return
    for i in range(0, len(pts_xy), max(1, step)):
        x, y = int(round(pts_xy[i,0])), int(round(pts_xy[i,1]))
        cv2.circle(img, (x,y), radius, color, -1)

def _to_img_pts(world_pts: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    if world_pts is None or len(world_pts) == 0:
        return None
    mask = np.isfinite(world_pts).all(axis=1)
    if not np.any(mask):
        return None
    wp = world_pts[mask]
    xs, ys = to_image_frame(wp, H)
    return np.stack([xs, ys], axis=1)

def normalize_prediction_res(prediction_res) -> Dict[int, np.ndarray]:
    out = {}
    if prediction_res is None:
        return out
    if isinstance(prediction_res, dict):
        if 'preds' in prediction_res:
            preds = prediction_res['preds']
            pids = prediction_res.get('pids', list(range(len(preds))))
            for i, pid in enumerate(pids):
                out[int(pid)] = np.asarray(preds[i])
        else:
            for k,v in prediction_res.items():
                try:
                    pid = int(k)
                except Exception:
                    continue
                out[pid] = np.asarray(v)
    else:
        arr = np.asarray(prediction_res)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            for i in range(arr.shape[0]):
                out[i] = arr[i]
    return out

class RawVideoOverlay:
    def __init__(self,
                 dataset: str,
                 out_video_path: str,
                 frame_offset: int = 0,
                 sim_dt: float = 0.1,
                 extracted_fps: float = 2.5,
                 output_fps: Optional[float] = None):
        self.asset_dir = FRAMES_ROOT
        self.dataset = dataset
        self.frames_dir = find_frames_dir(self.asset_dir, dataset)
        self.H = load_homography(HZIP_PATH, dataset)
        self.frame_offset = frame_offset
        self.sim_dt = sim_dt
        self.extracted_fps = extracted_fps
        self.video_stride = max(1, int(round((1.0/extracted_fps) / sim_dt)))  # e.g., 0.4 / 0.1 = 4
        self.raw_frame_index = 0
        self.sim_step = 0

        self.writer = None
        self.out_video_path = out_video_path
        self.output_fps = output_fps if output_fps is not None else extracted_fps

        self._probe_size()

    def _probe_size(self):
        pngs = glob.glob(os.path.join(self.frames_dir, "*.png"))
        if not pngs:
            raise FileNotFoundError(f"No PNG frames found under {self.frames_dir}. Run video_parser.py first.")
        h = w = None
        for p in pngs[:10]:
            img = cv2.imread(p)
            if img is not None:
                h, w = img.shape[:2]
                break
        if h is None or w is None:
            raise RuntimeError(f"Could not read any frame under {self.frames_dir}")
        self.size = (w, h)

    def _ensure_writer(self):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.out_video_path, fourcc, self.output_fps, self.size)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def should_render_this_step(self) -> bool:
        return (self.sim_step % self.video_stride) == 0

    def _frame_path_for_current(self) -> Optional[str]:
        fidx = self.raw_frame_index + self.frame_offset
        fpath = os.path.join(self.frames_dir, f"{fidx}.png")
        if os.path.isfile(fpath):
            return fpath
        return None

    def step(self,
             valid_obs: Dict[int, np.ndarray],
             valid_obs_future_true: Dict[int, np.ndarray],
             prediction_res) -> bool:
        wrote = False
        if self.should_render_this_step():
            frame_path = self._frame_path_for_current()
            if frame_path is not None:
                img = cv2.imread(frame_path)
                if img is not None:
                    for pid, hist in (valid_obs or {}).items():
                        hist = np.asarray(hist)
                        img_pts = _to_img_pts(hist, self.H)
                        if img_pts is not None:
                            _polyline(img, img_pts, color=(200,100,0), thickness=2)  # history
                    for pid, fut in (valid_obs_future_true or {}).items():
                        fut = np.asarray(fut)
                        img_pts = _to_img_pts(fut, self.H)
                        if img_pts is not None:
                            _dots(img, img_pts, color=(0,0,0), radius=2, step=1)   # GT
                    preds = normalize_prediction_res(prediction_res)
                    for pid, pred in preds.items():
                        pred = np.asarray(pred)
                        img_pts = _to_img_pts(pred, self.H)
                        if img_pts is not None:
                            _polyline(img, img_pts, color=(0,0,255), thickness=2)   # prediction
                            hx, hy = int(round(img_pts[-1,0])), int(round(img_pts[-1,1]))
                            cv2.circle(img, (hx,hy), 3, (0,0,255), -1)

                    self._ensure_writer()
                    self.writer.write(img)
                    wrote = True
            self.raw_frame_index += 1
        self.sim_step += 1
        return wrote
