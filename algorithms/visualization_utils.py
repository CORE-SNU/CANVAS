import os
import yaml
import pathlib
import numpy as np
import cv2
from matplotlib.patches import Polygon, Rectangle, Circle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
def draw_map4(xlim=15., ylim=10., belx=0., bely=0., bg_img_path=None):

    # Clear current figure and axes
    plt.clf(), plt.cla()
    fig, ax = plt.subplots(figsize=(8, 8))

    if bg_img_path is not None:
        try:
            filename = pathlib.Path(bg_img_path).name
            bg_image = plt.imread(bg_img_path)
            extent = [belx, xlim, bely, ylim]
            ax.imshow(bg_image, extent=extent, aspect='auto')
        except Exception as e:
            print(f"Error loading background image: {e}")

    ax.set_aspect('equal')
    ax.axis('on')
    
    return fig, ax
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

def draw_map2(xlim=15.,ylim=10.,belx=0.,bely=0.):
    plt.clf(), plt.cla()
    fig, ax = plt.subplots(figsize=(6, 6))
    map_metadata = load_map()
    # resolution: meters/pixel
    resolution = map_metadata['resolution']
    origin = np.array(map_metadata['origin'])
    image_path = pathlib.Path(map_metadata['image'])
    image = cv2.imread(str(image_path.resolve()), -1)
    h_pixel, w_pixel = image.shape



    xmin = origin[0]
    ymin = origin[1]

    xmax = origin[0] + w_pixel * resolution
    ymax = origin[1] + h_pixel * resolution
    ax.set_xlim(belx, xlim)
    ax.set_ylim(bely, ylim)

    ax.set_aspect('equal')
    ax.axis('on')

    return fig, ax
def draw_map():
    plt.clf(), plt.cla()
    fig, ax = plt.subplots(figsize=(6, 6))
    map_metadata = load_map()
    # resolution: meters/pixel
    resolution = map_metadata['resolution']
    origin = np.array(map_metadata['origin'])
    image_path = pathlib.Path(map_metadata['image'])
    image = cv2.imread(str(image_path.resolve()), -1)
    h_pixel, w_pixel = image.shape

    xmin = origin[0]
    ymin = origin[1]
    xmax = origin[0] + w_pixel * resolution
    ymax = origin[1] + h_pixel * resolution
    ax.imshow(image, extent=(xmin, xmax, ymin, ymax), cmap='gray', alpha=0.5)

    fc = '#e3d0bf'
    ec = 'none'
    visualize_polytope(BOUND, ax, fc, ec)

    fc = '#bcd4e6'
    ec = '#87acc7'
    for region_id, vertices in SPOTS.items():
        visualize_polytope(vertices, ax, fc, ec)

    fc = '#e4bec2'
    ec = '#c48a90'
    for obstacle_id, vertices in OBSTACLES.items():
        visualize_polytope(vertices, ax, fc, ec)

    ax.set_xlim(-7., 13.)
    ax.set_ylim(-12., 5.)

    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax



def load_map():
    with open(os.path.join(os.path.dirname(__file__), "lobby.yaml")) as f:
        map_metadata = yaml.safe_load(f)
        return map_metadata


def pixel2meter(pixel_x, pixel_y, resolution, origin, h_pixel):
    meter_x = origin[0] + resolution * pixel_x
    meter_y = origin[1] + resolution * (h_pixel - pixel_y)
    return meter_x, meter_y


def visualize_polytope(vertices, ax, facecolor, edgecolor):

    polygon = Polygon(vertices,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      alpha=0.5,
                      linewidth=1
                      )
    ax.add_patch(polygon)


def polytope_in_meter(vertices, resolution, origin, h_pixel):
    return np.array([pixel2meter(px, py, resolution, origin, h_pixel) for px, py in vertices])


MAP_METADATA = load_map()
# resolution: meters/pixel
RESOLUTION = MAP_METADATA['resolution']
ORIGIN = np.array(MAP_METADATA['origin'])
IMAGE_PATH = pathlib.Path(MAP_METADATA['image'])
IMAGE = cv2.imread(str(IMAGE_PATH.resolve()), -1)
H_PIXEL, W_PIXEL = IMAGE.shape


SPOTS_IN_PIXEL = {
    'stair': [(912, 827), (906, 884), (927, 888), (934, 834)],
    'elevator': [(945, 781), (943, 804), (993, 806), (995, 784)],
    'bathroom': [(1031, 785), (1019, 842), (995, 843), (1007, 787)],
    'corridor': [(853, 914), (846, 985), (887, 988), (897, 921)],
    'security_room': [(874, 1302), (869, 1398), (910, 1402), (913, 1306)],
    'faculty_lounge': [(1380, 1344), (1379, 1399), (1350, 1401), (1352, 1348)],
    'lounge': [(1473, 947), (1471, 975), (1579, 979), (1578, 954)],
    'office': [(1574, 993), (1576, 1041), (1543, 1041), (1546, 997)],
    'main_entrance': [(1086, 1311), (1086, 1332), (1173, 1335), (1175, 1314)],
    'sub_entrance': [(1009, 1315), (988, 1314), (988, 1359), (1010, 1360)]
}

SPOTS = {
    spot_id: polytope_in_meter(vertices, RESOLUTION, ORIGIN, H_PIXEL) for spot_id, vertices in SPOTS_IN_PIXEL.items()
}


SPAWN_FREQUENCIES = {
    'stair': 0.,
    'elevator': 0.,
    'bathroom': 0.,
    'corridor': 0.,
    'security_room': 0.,
    'faculty_lounge': 0.,
    'lounge': 0.,
    'office': 0.,
    'main_entrance': 0.,
    'sub_entrance': 0.
}


INIT_ANGLES = {
    'stair': 0.,
    'elevator': -np.pi / 2.,
    'bathroom': -np.pi / 2.,
    'corridor': 0.,
    'security_room': 0.,
    'faculty_lounge': np.pi,
    'lounge': -np.pi / 2.,
    'office': np.pi,
    'main_entrance': np.pi / 2.,
    'sub_entrance': np.pi
}

OBSTACLES_IN_PIXEL = {
    'pillar_left': [(1006, 1015), (992, 1015), (981, 1028), (985, 1053), (1008, 1056), (1021, 1042), (1018, 1022)],
    'pillar_right': [(1281, 1032), (1274, 1052), (1281, 1066), (1306, 1074), (1318, 1055), (1313, 1036)],
    'wall_north': [(1034, 785), (1004, 917), (1590, 955), (1588, 858)],
    'wall_east': [(1412, 1046), (1383, 1485), (1579, 1499), (1585, 1061)],
    'wall_south': [(865, 1403), (861, 1456), (1380, 1480), (1386, 1427)],
    'wall_west': [(794, 983), (779, 1416), (867, 1424), (888, 995)],
    'wall_northwest': [(833, 911), (906, 919), (919, 760), (846, 765)],
    'fanuc': [(1122, 1077), (1113, 1203), (1160, 1206), (1171, 1082)],
    'entrance_wall_left': [(1019, 1302), (1013, 1425), (1084, 1430), (1092, 1308)],
    'entrance_wall_right': [(1183, 1306), (1171, 1425), (1243, 1434), (1252, 1311)]
}

OBSTACLES = {
    spot_id: polytope_in_meter(vertices, RESOLUTION, ORIGIN, H_PIXEL) for spot_id, vertices in OBSTACLES_IN_PIXEL.items()
}


BOUND_IN_PIXEL = [(898, 777), (873, 1406), (1557, 1461), (1577, 937)]

BOUND = polytope_in_meter(BOUND_IN_PIXEL, RESOLUTION, ORIGIN, H_PIXEL)


def visualize_tracking_result(tracking_result, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = itertools.cycle(prop_cycle.by_key()['color'])
    colors = prop_cycle.by_key()['color']
    n_colors = len(colors)

    for obj_id, traj in tracking_result.items():
        traj_np = np.array(traj)
        color = colors[obj_id % n_colors]
        visualize_trajectory(traj_np, ax, color='#88729a')
        center = traj_np[-1]
        radius = 0.3
        circ = Circle(center, radius, facecolor='#88729a', edgecolor='tab:gray', zorder=90)
        ax.add_patch(circ)

        ax.text(center[0], center[1], '{}'.format(obj_id), fontsize=8, zorder=100)

    return


def visualize_trajectory(trajectory, ax, color='black'):
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=60)


def visualize_prediction_result(prediction_result, ax, color='k', linestyle='solid'):
    labeled = False
    for obj_id, t in prediction_result.items():
        if t is not None:
            if not labeled:
                ax.plot(t[:, 0], t[:, 1], zorder=80, linewidth=2, color=color, label='prediction', linestyle=linestyle)
                labeled = True
            else:
                ax.plot(t[:, 0], t[:, 1], zorder=80, linewidth=2, color=color, linestyle=linestyle)


def visualize_cp_result2(confidence_intervals, prediction_result, selected_steps, ax, bg_img=None, extent=None):
    n_selected = len(selected_steps)
    max_transparency = 0.6
    min_transparency = 0.3
    transparency_diff = max_transparency - min_transparency

    for obj_id, traj in prediction_result.items():
        for idx, step in enumerate(selected_steps):
            center = traj[step]
            radius = confidence_intervals[step]
            alpha = max_transparency - transparency_diff * idx / (n_selected - 1)

            # Draw the CP circle
            circ = Circle(center, radius=radius,
                          facecolor='tab:gray', alpha=alpha,
                          edgecolor='none', zorder=30)
            ax.add_patch(circ)

            # Annotate with step number
            ax.text(center[0], center[1], str(step),
                    fontsize=8, ha='center', va='center',
                    color='white', weight='bold', zorder=100)

            # Redraw background within the patch
            if bg_img is not None and extent is not None:
                ax.imshow(bg_img,
                          extent=extent,
                          clip_path=circ,
                          clip_on=True,
                          zorder=20)
    return

def visualize_cp_result(confidence_intervals, prediction_result, selected_steps, ax):
    n_predictions = confidence_intervals.size
    n_selected = len(selected_steps)
    max_transparency = 0.6
    min_transparency = 0.3
    transparency_diff = max_transparency - min_transparency
    for obj_id, t in prediction_result.items():
        count = 0
        for i in selected_steps:
            center = t[i]
            radius = confidence_intervals[i]
            transparency = max_transparency - transparency_diff * count / (n_selected - 1)
            circ = Circle(center, radius=radius, color='tab:gray', alpha=transparency, zorder=30)
            count += 1
            ax.add_patch(circ)
    return


def visualize_controller_info(info, ax):
    if info['feasible']:
        paths = info['candidate_paths']
        safe_paths = info['safe_paths']
        final_path = info['final_path']
        '''
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], color='tab:gray', zorder=60, alpha=0.1)

        for sp in safe_paths:
            ax.plot(sp[:, 0], sp[:, 1], color='yellow', zorder=70, alpha=0.2)
        '''
        ax.plot(final_path[:, 0], final_path[:, 1], color='tab:cyan', zorder=80)
    return
