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
        fname = "eth-ucy/biwi_eth.npy"
    elif key in ("hotel", "biwi_hotel", "biwi-hotel"):
        fname = "eth-ucy/biwi_hotel.npy"
    elif key in ("zara1", "zara01", "crowds_zara01", "zara_1"):
        fname = "eth-ucy/crowds_zara01.npy"
    elif key in ("zara2", "zara02", "crowds_zara02", "zara_2"):
        fname = "eth-ucy/crowds_zara02.npy"
    elif key in ("students001", "students1", "ucy_students1", "univ1"):
        fname = "eth-ucy/students001.npy"
    elif key in ("students003", "students3", "univ", "ucy_univ", "university"):
        fname = "eth-ucy/students003.npy"
    elif key in ("lobby3"):
        fname = "snu-asri/0.npy"
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

def _make_regions(name: str) -> List[Dict[str, float]]:
    if name == "snu-asri":
        return [
            {"name": "glass door below", "xmin": 0.3, "xmax": 5.0, "ymin": -12.0, "ymax": -6.3},
            {"name": "left glass", "xmin": -7.0, "xmax": 0.3, "ymin": -12.0, "ymax": -8.5},
            {"name": "right glass", "xmin": 5.0, "xmax": 13.0, "ymin": -12.0, "ymax": -8.5},
            {"name": "left wall", "xmin": -7.0, "xmax": -2.1, "ymin": -12.0, "ymax": -0.3},
            {"name": "right wall", "xmin": 7.8, "xmax": 13.0, "ymin": -12.0, "ymax": -0.3},
            {"name": "upper-left wall", "xmin": -7.0, "xmax": -1.9, "ymin": 0.9, "ymax": 5.0},
            {"name": "upper wall", "xmin": -0.5, "xmax": 13.0, "ymin": 0.9, "ymax": 5.0},
            {"name": "middle square", "xmin": 2.0, "xmax": 3.4, "ymin": -4.6, "ymax": -1.8},
            {"name": "left cylinder", "xmin": -0.8, "xmax": 0.2, "ymin": -1.5, "ymax": -0.7},
            {"name": "right cylinder", "xmin": 5.3, "xmax": 6.3, "ymin": -1.5, "ymax": -0.7},
        ]
    
    if name == "eth":
        return [
            {"name": "left road", "xmin": -10.0, "xmax": -5.2, "ymin": -7.0, "ymax": 18.0},
            {"name": "lower ground", "xmin": 0.0, "xmax": 13.0, "ymin": -7.0, "ymax": -1.0},
            {"name": "upper ground", "xmin": -1.5, "xmax": 13.0, "ymin": 13.0, "ymax": 18.0},
            {"name": "right building", "xmin": 13.0, "xmax": 20.0, "ymin": -7.0, "ymax": 18.0}
        ]
    
    if name == "hotel":
        return [
            {"name": "left road", "xmin": -3.0, "xmax": -0.8, "ymin": -11.0, "ymax": 5.0},
            {"name": "left bench", "xmin": 0.9, "xmax": 1.7, "ymin": -11.0, "ymax": -8.5},
            {"name": "left tree", "xmin": 1.2, "xmax": 1.8, "ymin": -6.3, "ymax": -5.7},
            {"name": "light", "xmin": 1.2, "xmax": 1.8, "ymin": -2.3, "ymax": -1.7},
            {"name": "right tree", "xmin": 1.2, "xmax": 1.8, "ymin": 1.7, "ymax": 2.3},
            {"name": "right building", "xmin": 4.2, "xmax": 7.0, "ymin": -11.0, "ymax": 5.0}
        ]
    
    if name == "Zara01" or name in ("zara1", "zara01", "crowds_zara01", "zara_1"):
        return [
            {"name": "left building", "xmin": 0.0, "xmax": 9.5, "ymin": 7.8, "ymax": 14.0},
            {"name": "left car", "xmin": 1.4, "xmax": 6.4, "ymin": 0.0, "ymax": 3.0},
            {"name": "right upper car", "xmin": 9.5, "xmax": 11.8, "ymin": 6.4, "ymax": 14.0}
        ]
    
    if name == "zara2":
        return [
            {"name": "left building", "xmin": -0.5, "xmax": 9.8, "ymin": 8.8, "ymax": 15.0},
            {"name": "right upper car", "xmin": 9.8, "xmax": 12.3, "ymin": 7.1, "ymax": 15.0}
        ]
    
    if name == "univ":
        return [
            {"name": "upper place", "xmin": 3.8, "xmax": 11.4, "ymin": 12.4, "ymax": 14.5},
            {"name": "upper down place", "xmin": 5.7, "xmax": 10.0, "ymin": 10.8, "ymax": 14.5},
            {"name": "left box", "xmin": -0.4, "xmax": 3.9, "ymin": 9.6, "ymax": 11.6}
        ]
    
    raise ValueError(f"Unknown dataset: {name}")

def get_dataset_spec(name: str) -> DatasetSpec:
    HERE = Path(__file__).resolve().parent
    if name == "snu-asri":
        bg = BgSpec(path=HERE/"lobby3.png", extent=(-3.0, 8.5, -9.5, 1.5), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions(name))

    if name == "eth":
        bg = BgSpec(path=HERE/"eth.png", extent=(-8.69, 18.42, -6.17, 17.21), rotate90=True, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions(name))

    if name == "hotel":
        bg = BgSpec(path=HERE/"hotel.png", extent=(-3.25, 6.35, -10.31, 4.31), rotate90=True, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions(name))

    if name == "zara1" or name in ("zara1", "zara01", "crowds_zara01", "zara_1"):
        bg = BgSpec(path=HERE/"crowds_zara01.jpg",
                    extent=(-0.02104651, 15.13244069, 0.76134018, 13.3864436), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions(name))

    if name == "zara2":
        bg = BgSpec(path=HERE/"crowds_zara02.jpg",
                    extent=(-0.357790686363, 15.558422764, 0.726257209729, 14.9427441591), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions(name))

    if name == "univ":
        bg = BgSpec(path=HERE/"students_003.jpg",
                    extent=(-0.174686040989, 15.4369843957, -0.222192273533, 13.8542013734), rotate90=False, alpha=0.6)
        return DatasetSpec(name=name, bg=bg, static_regions=_make_regions(name))

    raise ValueError(f"Unknown dataset: {name}")

