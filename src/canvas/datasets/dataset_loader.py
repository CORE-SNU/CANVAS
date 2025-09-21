import os
import yaml
import pathlib
import cv2
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
from pathlib import Path

_DATA_DIR = os.path.dirname(__file__)

def load_dataset(name_or_path: Union[str, os.PathLike]):
    """
    Load one of the ETH/UCY datasets by short code (e.g., 'ETH', 'ZARA1')
    or by giving a direct path to a .npy file.

    Supported codes (case-insensitive):
      - 'ETH'        -> biwi_eth.npy
      - 'HOTEL'      -> biwi_hotel.npy
      - 'ZARA1'      -> crowds_zara01.npy
      - 'ZARA2'      -> crowds_zara02.npy
      - 'STUDENTS001'-> students001.npy(alias: 'STUDENTS1')
      - 'STUDENTS003'-> students003.npy(aliases: 'UNIV', 'STUDENTS3')
      - '0'          -> 0.npy  (if you need that file)

    You can also pass a relative/absolute path like 'datasets/biwi_eth.npy'.
    """
    s = str(name_or_path).strip()

    # If it looks like a file path or ends with .npy, load it directly.
    if os.path.sep in s or s.lower().endswith(".npy"):
        path = s if os.path.isabs(s) else os.path.join(_DATA_DIR, s)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return np.load(path)

    # Otherwise, interpret as a dataset code.
    key = s.lower()

    if key in ("eth", "biwi_eth", "biwi-eth"):
        fname = "biwi_eth.npy"
    elif key in ("hotel", "biwi_hotel", "biwi-hotel"):
        fname = "biwi_hotel.npy"
    elif key in ("zara1", "zara01", "crowds_zara01", "zara_1"):
        fname = "crowds_zara01.npy"
    elif key in ("zara2", "zara02", "crowds_zara02", "zara_2"):
        fname = "crowds_zara02.npy"
    elif key in ("students001", "students1", "ucy_students1", "univ1"):
        fname = "students001.npy"
    elif key in ("students003", "students3", "univ", "ucy_univ", "university"):
        fname = "students003.npy"
    elif key in ("lobby3"):
        fname = "0.npy"
    else:
        raise ValueError(
            f"Unknown dataset code '{name_or_path}'. "
            "Use one of: ETH, HOTEL, ZARA1, ZARA2, STUDENTS001, STUDENTS003 (UNIV), "
            "or provide a direct .npy path."
        )

    path = os.path.join(_DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return np.load(path)

@dataclass
class BgSpec:
    path: Path
    extent: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    rotate90: bool = False
    alpha: float = 0.6

@dataclass
class DatasetSpec:
    name: str
    bg: BgSpec
    static_regions: List[Dict[str, Any]]

def _load_background_image(pathlike, rotate90: bool):
    img = cv2.imread(str(pathlike), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(pathlike)
    if rotate90:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 90, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    # BGR → RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _make_regions() -> List[Dict[str, float]]:
    return [
        {"name": "glass door below", "xmin": 0.3, "xmax": 5.0, "ymin": -12.0, "ymax": -6.3},
        {"name": "left glass", "xmin": -7.0, "xmax": 0.3, "ymin": -12.0, "ymax": -8.5},
        {"name": "right glass", "xmin": 5.0, "xmax": 13.0, "ymin": -12.0, "ymax": -8.5},
        {"name": "left wall", "xmin": -7.0, "xmax": -2.1, "ymin": -12.0, "ymax": -0.3},
        {"name": "right wall", "xmin": 7.8, "xmax": 13.0, "ymin": -12.0, "ymax": -0.3},
        {"name": "upper-left wall", "xmin": -7.0, "xmax": -1.9, "ymin": 1.1, "ymax": 5.0},
        {"name": "upper wall", "xmin": -0.5, "xmax": 13.0, "ymin": 0.9, "ymax": 5.0},
        {"name": "middle square", "xmin": 2.0, "xmax": 3.4, "ymin": -4.6, "ymax": -1.6},
        {"name": "left cylinder", "xmin": -0.7, "xmax": 0.5, "ymin": -1.5, "ymax": -0.6},
        {"name": "right cylinder", "xmin": 5.3, "xmax": 6.4, "ymin": -1.8, "ymax": -0.8},
    ]

def get_dataset_spec(name: str) -> DatasetSpec:
    HERE = Path(__file__).resolve().parent
    if name == "Lobby":
        bg = BgSpec(path=HERE/"lobby3.png", extent=(-3.0, 8.5, -9.5, 1.5), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions())

    if name == "ETH":
        bg = BgSpec(path=HERE/"eth.png", extent=(-8.69, 18.42, -6.17, 17.21), rotate90=True, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=[])

    if name == "Hotel":
        bg = BgSpec(path=HERE/"hotel.png", extent=(-3.25, 6.35, -10.31, 4.31), rotate90=True, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=[])

    if name == "Zara01":
        bg = BgSpec(path=HERE/"crowds_zara01.jpg",
                    extent=(-0.02104651, 15.13244069, 0.76134018, 13.3864436), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=[])

    if name == "Zara02":
        bg = BgSpec(path=HERE/"crowds_zara02.jpg",
                    extent=(-0.357790686363, 15.558422764, 0.726257209729, 14.9427441591), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=[])

    if name == "Univ":
        bg = BgSpec(path=HERE/"students_003.jpg",
                    extent=(-0.174686040989, 15.4369843957, -0.222192273533, 13.8542013734), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=[])

    raise ValueError(f"Unknown dataset: {name}")

'''
def load_map():
    with open(os.path.join(os.path.dirname(__file__), "lobby.yaml")) as f:
        map_metadata = yaml.safe_load(f)
        return map_metadata

def draw_map3(xlim=15., ylim=10., belx=0., bely=0., bg_img_path=None):

    # Clear current figure and axes
    plt.clf(), plt.cla()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Load map metadata and the map image used for an overlay (if needed)
    map_metadata = load_map()
    resolution = map_metadata['resolution']
    origin = np.array(map_metadata['origin'])
    image_path = pathlib.Path(map_metadata['image'])
    image = cv2.imread(str(image_path.resolve()), -1)
    h_pixel, w_pixel = image.shape

    # Calculate the map boundaries (may be used for overlay)
    xmin = origin[0]
    ymin = origin[1]
    xmax = origin[0] + w_pixel * resolution
    ymax = origin[1] + h_pixel * resolution

    # Set the plot limits based on the provided parameters
    ax.set_xlim(belx, xlim)
    ax.set_ylim(bely, ylim)

    # If a background image path is provided, load and display it
    if bg_img_path is not None:
        try:
            filename = pathlib.Path(bg_img_path).name
            # Process eth.png: rotate by 90 degrees with specific coordinates
            if filename == "eth.png":
                img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
                angle = 90  # Rotate by 90 degrees
                (h, w) = img.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, M, (w, h))
                if rotated_img.ndim == 3 and rotated_img.shape[2] == 3:
                    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
                extent = [-8.69, 18.42, -6.17, 17.21]
                ax.imshow(rotated_img, extent=extent,alpha=0.4, aspect='auto', zorder=0)
            # Process hotel.png similarly: rotate by 90 degrees and use custom coordinates
            elif filename == "hotel.png":
                img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
                angle = 90  # Rotate by 90 degrees
                (h, w) = img.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, M, (w, h))
                if rotated_img.ndim == 3 and rotated_img.shape[2] == 3:
                    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
                extent = [-3.25, 6.35, -10.31, 4.31]
                ax.imshow(rotated_img, extent=extent, alpha=0.6,aspect='auto')
            # Custom coordinates for crowds_zara01.jpg
            elif filename == "crowds_zara01.jpg":
                bg_image = plt.imread(bg_img_path)
                extent = [-0.02104651, 15.13244069, 0.76134018, 13.3864436]
                rotated_img=bg_image
                ax.imshow(bg_image, extent=extent,alpha=0.6, aspect='auto')
            # Custom coordinates for crowds_zara02.jpg
            elif filename == "crowds_zara02.jpg":
                bg_image = plt.imread(bg_img_path)
                extent = [-0.357790686363, 15.558422764, 0.726257209729, 14.9427441591]
                rotated_img=bg_image
                ax.imshow(bg_image, extent=extent,alpha=0.6, aspect='auto')
            # Custom coordinates for students_003.jpg
            elif filename == "students_003.jpg":
                bg_image = plt.imread(bg_img_path)
                extent = [-0.174686040989, 15.4369843957, -0.222192273533, 13.8542013734]
                rotated_img=bg_image
                ax.imshow(bg_image, extent=extent,alpha=0.6, aspect='auto')
            else:
                # Default: use the plot limits provided if no special coordinates are defined
                bg_image = plt.imread(bg_img_path)
                extent = [belx, xlim, bely, ylim]
                rotated_img=bg_image
                ax.imshow(bg_image, extent=extent,alpha=0.6, aspect='auto')
        except Exception as e:
            print(f"Error loading background image: {e}")

    # Optionally, overlay the original map image (if needed) with a higher z-order:
    # ax.imshow(image, extent=[xmin, xmax, ymin, ymax], cmap='gray', zorder=1)

    ax.set_aspect('equal')
    ax.axis('on')
    
    return fig, ax,rotated_img,extent
'''
