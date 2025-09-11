import os
import numpy as np
from typing import Union

# Directory that contains the .npy files shown in your screenshot
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
      - 'STUDENTS001' (alias: 'STUDENTS1') -> students001.npy
      - 'STUDENTS003' (aliases: 'UNIV', 'STUDENTS3') -> students003.npy
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
