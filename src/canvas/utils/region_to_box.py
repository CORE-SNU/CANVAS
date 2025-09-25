import numpy as np
from ..detection.detection_utils import Box
def region_to_box(region: dict, default_deg: float = 0.0, resolution: float = 1e-3) -> Box:
    xmin, xmax = region["xmin"], region["xmax"]
    ymin, ymax = region["ymin"], region["ymax"]
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    deg = float(region.get("deg", default_deg))
    rad = radians(deg)
    
    corners = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2],
    ], dtype=float)
    
    c, s = cos(rad), sin(rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    rot_corners = (corners @ R.T) + np.array([x_center, y_center])
    return Box(
        x=x_center, y=y_center, w=w, h=h,
        deg=deg, rad=rad, area=w*h,
        vertices=rot_corners,
        resolution=resolution,
        pos=np.array([x_center, y_center], dtype=float)
    )
