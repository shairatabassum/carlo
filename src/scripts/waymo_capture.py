import argparse
import carla
import random
import os
import numpy as np
import cv2
import open3d as o3d
import time
import json
import imageio

from src.common.spawn import spawn_ego, spawn_vehicles
from src.sensors.lidar import Lidar, LidarSettings

SAVE_INTERVAL = 10    # save every Nth tick
NUM_VEHICLES = 0   # number of vehicles to spawn
MAX_FRAMES = 200    # number of saved frames
MIN_TRACK_FRAMES = 10  # discard vehicles visible in fewer frames

CAMERA_W = 1920
CAMERA_H = 1280
CAMERA_FOV = 60

OUTPUT_DIR = "./runs_waymo/nikt_paper/conf8/"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
EGO_DIR = os.path.join(OUTPUT_DIR, "ego_pose")
EXTRINSIC_DIR = os.path.join(OUTPUT_DIR, "extrinsics")                    # StreetGaussian / EmerNeRF
EXTRINSIC_DESIREGS_DIR = os.path.join(OUTPUT_DIR, "extrinsics_desiregs")  # DeSiRe-GS
INTRINSIC_DIR = os.path.join(OUTPUT_DIR, "intrinsics")
TRACK_DIR = os.path.join(OUTPUT_DIR, "track")
DYNAMIC_MASK_DIR = os.path.join(OUTPUT_DIR, "dynamic_mask")
SKY_MASK_DIR = os.path.join(OUTPUT_DIR, "sky_mask")
LIDAR_DIR        = os.path.join(OUTPUT_DIR, "lidar")         # DeSiRe-GS  Nx10 .bin
LIDAR_EMERNERF_DIR = os.path.join(OUTPUT_DIR, "lidar_emernerf")  # EmerNeRF   Nx14 .bin
NORMAL_DIR       = os.path.join(OUTPUT_DIR, "normals")       # DeSiRe-GS surface normals
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EGO_DIR, exist_ok=True)
os.makedirs(EXTRINSIC_DIR, exist_ok=True)
os.makedirs(EXTRINSIC_DESIREGS_DIR, exist_ok=True)
os.makedirs(INTRINSIC_DIR, exist_ok=True)
os.makedirs(TRACK_DIR, exist_ok=True)
os.makedirs(LIDAR_DIR, exist_ok=True)
os.makedirs(LIDAR_EMERNERF_DIR, exist_ok=True)
os.makedirs(DYNAMIC_MASK_DIR, exist_ok=True)
os.makedirs(SKY_MASK_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)

# CARLA semantic label for sky
SKY_LABEL = 13

# CAMERA_NAMES = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
# CAMERA_NAMES = ["FRONT", "FRONT_LEFT_45", "FRONT_RIGHT_45", "SIDE_LEFT_90", "SIDE_RIGHT_90"]
CAMERA_NAMES = ["FRONT", "LEFT", "RIGHT", "DOWN"]

# 12 edges of a 3D bounding box given corners ordered by (sx,sy,sz) in (1,-1)x(1,-1)x(1,-1)
BBOX_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

# carla to waymo-vehicle
carla_to_waymo_vehicle = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0,  1, 0],
    [0,  0,  0, 1],
])

opencv2camera = np.array([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.], # Maps Cam-X to Vehicle-Left
    [0., -1., 0., 0.], # Maps Cam-Y to Vehicle-Down (Negative Up)
    [0., 0., 0., 1.]
])

def carla_transform_to_matrix(transform):
    loc = transform.location
    rot = transform.rotation

    cy = np.cos(np.radians(rot.yaw))
    sy = np.sin(np.radians(rot.yaw))
    cp = np.cos(np.radians(rot.pitch))
    sp = np.sin(np.radians(rot.pitch))
    cr = np.cos(np.radians(rot.roll))
    sr = np.sin(np.radians(rot.roll))

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [loc.x, loc.y, loc.z]

    return T

def get_intrinsics(camera):
    width = int(camera.attributes["image_size_x"])
    height = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])

    fx = width / (2 * np.tan(np.radians(fov) / 2))
    fy = fx

    cx = width / 2.0
    cy = height / 2.0

    # CARLA has ideal pinhole cameras
    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0

    intrinsic = np.array([
        fx,
        fy,
        cx,
        cy,
        k1,
        k2,
        p1,
        p2,
        k3
    ])

    return intrinsic




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Consider this like a flag where you specify a seed if you want to keep the trajectory same every time you run. "
             "Pass --seed -1 for a different trajectory each run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = None if args.seed < 0 else args.seed
    if seed is not None:
        random.seed(seed)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    # -------------------------
    # Enable synchronous mode
    # -------------------------
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()


    # -------------------------
    # Spawn ego vehicle
    # -------------------------
    ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    # ego_transform = random.choice(spawn_points)
    ego_transform = carla.Transform(carla.Location(x=106.386559, y=-2.362594, z=0.5), carla.Rotation(pitch=0, yaw=-90, roll=0))
    print(f"Ego transform: {ego_transform}")
    ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    if seed is not None:
        traffic_manager.set_random_device_seed(seed)
    ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0)


    # -------------------------
    # Spawn NPC vehicles
    # -------------------------
    vehicle_bps = [
    bp for bp in blueprint_library.filter('vehicle.*') 
    if bp.id != 'vehicle.carlamotors.carlacola'
]

    vehicles = []
    for i in range(NUM_VEHICLES):
        try:
            bp = random.choice(vehicle_bps)
            transform = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                vehicles.append(vehicle)
        except:
            pass


    # -------------------------
    # RGB Cameras
    # -------------------------
    # camera_transforms = [
    #     ("FRONT", carla.Transform(carla.Location(x=1.5, z=1.6))),
    #     ("FRONT_LEFT", carla.Transform(carla.Location(x=1.5, y=-0.5, z=1.6), carla.Rotation(yaw=-45))),
    #     ("FRONT_RIGHT", carla.Transform(carla.Location(x=1.5, y=0.5, z=1.6), carla.Rotation(yaw=45))),
    #     ("SIDE_LEFT", carla.Transform(carla.Location(y=-0.8, z=1.6), carla.Rotation(yaw=-90))),
    #     ("SIDE_RIGHT", carla.Transform(carla.Location(y=0.8, z=1.6), carla.Rotation(yaw=90))),
    # ]
    # camera_transforms = [
    #     ("FRONT", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=0))),
    #     ("FRONT_LEFT_45", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-45))),
    #     ("FRONT_RIGHT_45", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=45))),
    #     ("SIDE_LEFT_90", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-90))),
    #     ("SIDE_RIGHT_90", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=90))),
    # ]
    
    camera_transforms = [
        ("FRONT", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=0))),
        ("LEFT", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
        ("RIGHT", carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-50))),
        ("DOWN", carla.Transform(carla.Location(z=3.0), carla.Rotation(pitch=-20)))
    ]

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAMERA_W))
    camera_bp.set_attribute("image_size_y", str(CAMERA_H))
    camera_bp.set_attribute("fov", str(CAMERA_FOV))

    cameras = []
    image_buffers = {}
    
    def make_camera_callback(cam_id):
        def callback(image):
            image_buffers[cam_id] = image
        return callback

    for cam_id, (_, transform) in enumerate(camera_transforms):
        cam = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)
        cam.listen(make_camera_callback(cam_id))
        cameras.append(cam)
        image_buffers[cam_id] = None

    # -------------------------
    # Semantic segmentation cameras (same pose as RGB, for sky mask)
    # -------------------------
    sem_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(CAMERA_W))
    sem_bp.set_attribute("image_size_y", str(CAMERA_H))
    sem_bp.set_attribute("fov", str(CAMERA_FOV))

    sem_cameras = []
    sem_buffers = {}

    def make_sem_callback(cam_id):
        def callback(image):
            sem_buffers[cam_id] = image
        return callback

    for cam_id, (_, transform) in enumerate(camera_transforms):
        sem_cam = world.spawn_actor(sem_bp, transform, attach_to=ego_vehicle)
        sem_cam.listen(make_sem_callback(cam_id))
        sem_cameras.append(sem_cam)
        sem_buffers[cam_id] = None

    # -------------------------
    # Normal cameras
    # Output: RGB JPG where pixel = (normal_camera_space + 1) / 2 * 255
    # -------------------------
    norm_bp = blueprint_library.find("sensor.camera.normals")
    norm_bp.set_attribute("image_size_x", str(CAMERA_W))
    norm_bp.set_attribute("image_size_y", str(CAMERA_H))
    norm_bp.set_attribute("fov", str(CAMERA_FOV))

    norm_cameras = []
    norm_buffers = {}

    def make_norm_callback(cam_id):
        def callback(image):
            norm_buffers[cam_id] = image
        return callback

    for cam_id, (_, transform) in enumerate(camera_transforms):
        norm_cam = world.spawn_actor(norm_bp, transform, attach_to=ego_vehicle)
        norm_cam.listen(make_norm_callback(cam_id))
        norm_cameras.append(norm_cam)
        norm_buffers[cam_id] = None

    # -------------------------
    # EXTRINSIC and INTRINSIC
    # -------------------------
    extrinsics_mat = {}   # cam_id -> 4x4, OpenCV cam -> Waymo vehicle
    intrinsics_mat = {}   # cam_id -> 3x3 K matrix
    inv_extrinsics = {}   # cam_id -> 4x4, Waymo vehicle -> OpenCV cam

    for cam_id, ((cam_name, cam_transform), cam_actor) in enumerate(zip(camera_transforms, cameras)):
        # camera --> vehicle (forward right up)
        cam_to_vehicle_carla = carla_transform_to_matrix(cam_transform)

        # convert CARLA vehicle frame to StreetGaussian/Waymo vehicle frame
        # x forward, y right, z up ----to---- x forward, y left, z up
        cam_to_vehicle_waymo = carla_to_waymo_vehicle @ cam_to_vehicle_carla

        # convert vehicle-frame camera into OpenCV camera convention (for StreetGaussian / EmerNeRF)
        extrinsic = cam_to_vehicle_waymo @ opencv2camera

        np.savetxt(os.path.join(EXTRINSIC_DIR, f"{cam_id}.txt"), extrinsic, fmt="%.18e")
        extrinsics_mat[cam_id] = extrinsic
        inv_extrinsics[cam_id] = np.linalg.inv(extrinsic)

        # DeSiRe-GS expects Waymo-cam → Waymo-ego (it applies OPENCV2DATASET internally).
        # Conjugating by carla_to_waymo_vehicle converts CARLA-cam axes to Waymo-cam axes
        # without the extra OpenCV rotation that StreetGaussian absorbs differently.
        extrinsic_desiregs = cam_to_vehicle_waymo @ carla_to_waymo_vehicle
        np.savetxt(os.path.join(EXTRINSIC_DESIREGS_DIR, f"{cam_id}.txt"), extrinsic_desiregs, fmt="%.18e")

        intrinsic_vec = get_intrinsics(cam_actor)
        np.savetxt(os.path.join(INTRINSIC_DIR, f"{cam_id}.txt"), intrinsic_vec, fmt="%.18e")
        fx, fy, cx, cy = intrinsic_vec[:4]
        intrinsics_mat[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


    # -------------------------
    # LiDAR
    # -------------------------
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "100")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", "460800")

    lidar_transform = carla.Transform(carla.Location(z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    
    lidar_latest = {"points": None}

    def lidar_callback(data):
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        lidar_latest["points"] = pts

    lidar.listen(lidar_callback)
    
    
    # -------------------------
    # Open3D LiDAR visualization
    # -------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR", width=960, height=540)

    pcd = o3d.geometry.PointCloud()
    pcd_added = False

    render_option = vis.get_render_option()
    render_option.point_size = 1.5
    render_option.background_color = np.array([0, 0, 0])  # black background

    ctr = vis.get_view_control()
    ctr.set_front([0, -1, -0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.3)


    # -------------------------
    # Main loop
    # -------------------------
    frame = 0      # raw tick counter
    save_idx = 0   # sequential index for file naming (000000, 000001, ...)
    timestamps = {"FRAME": {}}
    for name in CAMERA_NAMES:
        timestamps[name] = {}

    pts_3d_all = {}   # frame -> Nx3 float, Waymo vehicle frame
    pts_2d_all = {}   # frame -> Nx6 int16, [cam, px, py, cam2, px2, py2]

    lidar_z_offset = 1.8  # lidar sensor height above vehicle origin

    # Track — all lines buffered by track_id, filtered at end by MIN_TRACK_FRAMES
    track_lines_by_id = {}   # int track_id -> list of (frame, line_str)
    bbox_visible_dict = {}   # str(track_id) -> {str(frame_id): [cam_ids]}
    object_ids = {}          # str(actor_id) -> sequential int track_id
    track_vis_imgs = []      # list of RGB frames for track_vis.mp4

    # Warm-up tick: sensors need one tick before their buffers are populated
    world.tick()

    try:
        while save_idx < MAX_FRAMES:
            world.tick()
            
            # ---------------------
            # RGB cameras visualization
            # ---------------------
            images = []
            for cam_id in image_buffers:
                img = image_buffers[cam_id]
                if img is None:
                    continue

                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                array = array.reshape((img.height, img.width, 4))[:, :, :3]
                images.append(array)

            if len(images) == 5:
                vis_img = cv2.hconcat(images)
                cv2.imshow("Cameras", vis_img)
                cv2.waitKey(1)
                
            # ---------------------
            # LiDAR visualization
            # ---------------------
            pts = lidar_latest["points"]

            if pts is not None:
                xyz = pts[:, :3]

                # Normalize height → jet colormap
                z = xyz[:, 2]
                z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

                colors = np.zeros((xyz.shape[0], 3))
                colors[:, 0] = np.clip(1.5 - np.abs(z_norm - 0.75) * 4, 0, 1)  # red
                colors[:, 1] = np.clip(1.5 - np.abs(z_norm - 0.5)  * 4, 0, 1)  # green
                colors[:, 2] = np.clip(1.5 - np.abs(z_norm - 0.25) * 4, 0, 1)  # blue

                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if not pcd_added:
                    vis.add_geometry(pcd)
                    pcd_added = True

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                
                # time.sleep(0.005)
                
            
            # ---------------------
            # Save data
            # ---------------------
            if frame % SAVE_INTERVAL == 0:
                print(f"Saving frame {save_idx} (tick {frame})")

                t = frame * settings.fixed_delta_seconds
                frame_key = f"{save_idx:06d}"

                # save ego pose
                vehicle_world_carla = carla_transform_to_matrix(ego_vehicle.get_transform())
                vehicle_world_waymo = carla_to_waymo_vehicle @ vehicle_world_carla @ np.linalg.inv(carla_to_waymo_vehicle)
                np.savetxt(os.path.join(EGO_DIR, f"{frame_key}.txt"), vehicle_world_waymo, fmt="%.18e")
                timestamps["FRAME"][frame_key] = t

                # per-camera ego_pose = vehicle world pose at camera shutter time
                for cam_id, (cam_name, _) in enumerate(camera_transforms):
                    np.savetxt(os.path.join(EGO_DIR, f"{frame_key}_{cam_id}.txt"), vehicle_world_waymo, fmt="%.18e")
                    timestamps[cam_name][frame_key] = t

                # save LiDAR pointcloud and camera projections
                pts_raw = lidar_latest["points"]
                if pts_raw is not None:
                    xyz_s = pts_raw[:, :3]
                    intensity = pts_raw[:, 3]   # CARLA provides intensity as 4th channel
                    # sensor frame (x fwd, y right, z up) + z offset → Waymo vehicle (x fwd, y left, z up)
                    xyz_v = np.stack([
                        xyz_s[:, 0],
                        -xyz_s[:, 1],
                        xyz_s[:, 2] + lidar_z_offset,
                    ], axis=1)
                    pts_3d_all[save_idx] = xyz_v

                    # Shared lidar columns
                    N = xyz_v.shape[0]
                    origins_col  = np.tile([0.0, 0.0, lidar_z_offset], (N, 1))
                    ground_col   = (xyz_v[:, 2] < 0.3).astype(np.float32)
                    zeros_N      = np.zeros(N, dtype=np.float32)

                    # DeSiRe-GS: Nx10  origins(3)|points(3)|ground(1)|intensity(1)|elongation(1)|laser_id(1)
                    lidar_10 = np.column_stack([
                        origins_col, xyz_v, ground_col, intensity, zeros_N, zeros_N,
                    ]).astype(np.float32)
                    lidar_10.tofile(os.path.join(LIDAR_DIR, f"{save_idx:03d}.bin"))

                    # EmerNeRF: Nx14  origins(3)|points(3)|flow_xyz(3)|flow_class(1)|ground(1)|intensity(1)|elongation(1)|laser_id(1)
                    lidar_14 = np.column_stack([
                        origins_col, xyz_v,
                        np.zeros((N, 3)),        # flow_xyz  (unknown)
                        np.full(N, -1.0),         # flow_class (-1 = no label)
                        ground_col, intensity, zeros_N, zeros_N,
                    ]).astype(np.float32)
                    lidar_14.tofile(os.path.join(LIDAR_EMERNERF_DIR, f"{save_idx:03d}.bin"))

                    N = xyz_v.shape[0]
                    pts_2d = np.full((N, 6), -1, dtype=np.int16)
                    pts_h = np.concatenate([xyz_v, np.ones((N, 1))], axis=1)  # Nx4

                    for cam_id in range(len(camera_transforms)):
                        K = intrinsics_mat[cam_id]
                        pts_cam = (inv_extrinsics[cam_id] @ pts_h.T).T  # Nx4

                        in_front = pts_cam[:, 2] > 0
                        if not in_front.any():
                            continue

                        p = pts_cam[in_front, :3]
                        px = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2])
                        py = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2])
                        in_bounds = (px >= 0) & (px < CAMERA_W) & (py >= 0) & (py < CAMERA_H)

                        idx = np.where(in_front)[0][in_bounds]
                        ipx = px[in_bounds].astype(np.int16)
                        ipy = py[in_bounds].astype(np.int16)

                        # fill first free projection slot per point
                        slot1_free = pts_2d[idx, 0] == -1
                        i1 = idx[slot1_free]
                        pts_2d[i1, 0] = cam_id
                        pts_2d[i1, 1] = ipx[slot1_free]
                        pts_2d[i1, 2] = ipy[slot1_free]

                        slot2_free = (~slot1_free) & (pts_2d[idx, 3] == -1)
                        i2 = idx[slot2_free]
                        pts_2d[i2, 3] = cam_id
                        pts_2d[i2, 4] = ipx[slot2_free]
                        pts_2d[i2, 5] = ipy[slot2_free]

                    pts_2d_all[save_idx] = pts_2d

                # --- Track: bounding boxes + camera visibility + visualization ---
                ego_world_mat = carla_transform_to_matrix(ego_vehicle.get_transform())
                ego_world_inv = np.linalg.inv(ego_world_mat)
                vis_boxes_per_cam = {0: [], 1: [], 2: []}  # FRONT, FRONT_LEFT, FRONT_RIGHT
                dyn_masks = {c: np.zeros((CAMERA_H, CAMERA_W), dtype=np.uint8)
                             for c in range(len(camera_transforms))}

                for npc in world.get_actors().filter("vehicle.*"):
                    if npc.id == ego_vehicle.id:
                        continue

                    actor_key = str(npc.id)
                    if actor_key not in object_ids:
                        object_ids[actor_key] = len(object_ids)
                    track_id = object_ids[actor_key]

                    bb = npc.bounding_box
                    ex, ey, ez = bb.extent.x, bb.extent.y, bb.extent.z
                    bx, by, bz = bb.location.x, bb.location.y, bb.location.z

                    # 8 corners in NPC local frame (CARLA: x fwd, y right, z up)
                    signs = np.array([[sx, sy, sz]
                                      for sx in (1, -1)
                                      for sy in (1, -1)
                                      for sz in (1, -1)], dtype=np.float64)
                    corners_npc = signs * np.array([ex, ey, ez]) + np.array([bx, by, bz])
                    corners_npc_h = np.concatenate([corners_npc, np.ones((8, 1))], axis=1)

                    # NPC local → world → ego vehicle (CARLA convention)
                    npc_world_mat = carla_transform_to_matrix(npc.get_transform())
                    corners_ego = (ego_world_inv @ (npc_world_mat @ corners_npc_h.T)).T  # (8,4)

                    # Flip y: CARLA ego frame → Waymo vehicle frame
                    corners_ego[:, 1] *= -1

                    # Box center in Waymo ego frame
                    center_h = npc_world_mat @ np.array([bx, by, bz, 1.0])
                    center_ego = ego_world_inv @ center_h
                    tx, ty, tz = center_ego[0], -center_ego[1], center_ego[2]

                    # Heading in Waymo convention (yaw around z, CCW positive)
                    npc_ego_carla = ego_world_inv @ npc_world_mat
                    npc_ego_waymo = carla_to_waymo_vehicle @ npc_ego_carla @ carla_to_waymo_vehicle
                    heading = np.arctan2(npc_ego_waymo[1, 0], npc_ego_waymo[0, 0])

                    # Dimensions (full extents)
                    length, width, height = 2 * ex, 2 * ey, 2 * ez

                    # Horizontal speed
                    vel = npc.get_velocity()
                    speed = np.sqrt(vel.x**2 + vel.y**2)

                    # Compute 2D corners for all cameras first
                    track_key = str(track_id)
                    npc_corners_2d = {}  # cam_id -> (8,2), -1 if corner behind/outside camera
                    cam_visible = []
                    for cam_id in range(len(camera_transforms)):
                        K_cam = intrinsics_mat[cam_id]
                        pts_cam = (inv_extrinsics[cam_id] @ corners_ego.T).T  # (8,4)
                        corners_2d = np.full((8, 2), -1.0)
                        visible = False
                        for j in range(8):
                            if pts_cam[j, 2] > 0:
                                px = K_cam[0, 0] * pts_cam[j, 0] / pts_cam[j, 2] + K_cam[0, 2]
                                py = K_cam[1, 1] * pts_cam[j, 1] / pts_cam[j, 2] + K_cam[1, 2]
                                corners_2d[j] = [px, py]
                                if 0 <= px < CAMERA_W and 0 <= py < CAMERA_H:
                                    visible = True
                        if visible:
                            cam_visible.append(cam_id)
                        npc_corners_2d[cam_id] = corners_2d

                    # Only record track data for vehicles visible in at least one camera
                    if len(cam_visible) == 0:
                        continue

                    if track_id not in track_lines_by_id:
                        track_lines_by_id[track_id] = []
                    track_lines_by_id[track_id].append((save_idx,
                        f"{save_idx} {track_id} vehicle -10 "
                        f"{height:.4f} {width:.4f} {length:.4f} "
                        f"{tx:.4f} {ty:.4f} {tz:.4f} "
                        f"{heading:.6f} {speed:.4f}\n"
                    ))

                    if track_key not in bbox_visible_dict:
                        bbox_visible_dict[track_key] = {}
                    bbox_visible_dict[track_key][str(save_idx)] = sorted(cam_visible)

                    for cam_id in [0, 1, 2]:
                        vis_boxes_per_cam[cam_id].append(npc_corners_2d[cam_id])

                    # Dynamic mask: fill projected box for moving objects
                    if speed > 1.0:
                        for cam_id in range(len(camera_transforms)):
                            corners_2d = npc_corners_2d[cam_id]
                            valid_pts = corners_2d[corners_2d[:, 0] >= 0]
                            if len(valid_pts) >= 3:
                                hull = cv2.convexHull(valid_pts.astype(np.float32))
                                cv2.fillConvexPoly(dyn_masks[cam_id], hull.astype(np.int32), 255)

                # Build track_vis frame: FRONT_LEFT | FRONT | FRONT_RIGHT
                vis_frame_parts = []
                for cam_id in [1, 0, 2]:
                    img_data = image_buffers[cam_id]
                    if img_data is None:
                        break
                    arr = np.frombuffer(img_data.raw_data, dtype=np.uint8) \
                            .reshape((img_data.height, img_data.width, 4))[:, :, :3].copy()
                    for corners_2d in vis_boxes_per_cam[cam_id]:
                        for i, j in BBOX_EDGES:
                            if corners_2d[i, 0] >= 0 and corners_2d[j, 0] >= 0:
                                p1 = (int(corners_2d[i, 0]), int(corners_2d[i, 1]))
                                p2 = (int(corners_2d[j, 0]), int(corners_2d[j, 1]))
                                cv2.line(arr, p1, p2, (0, 255, 0), 2)
                    vis_frame_parts.append(arr[:, :, ::-1])  # BGR -> RGB for imageio
                if len(vis_frame_parts) == 3:
                    track_vis_imgs.append(np.concatenate(vis_frame_parts, axis=1))

                for cam_id in image_buffers:
                    img = image_buffers[cam_id]
                    if img is None:
                        continue

                    array = np.frombuffer(img.raw_data, dtype=np.uint8)
                    array = array.reshape((img.height, img.width, 4))[:, :, :3]

                    filename = f"{save_idx:06d}_{cam_id}.png"
                    path = os.path.join(IMAGE_DIR, filename)

                    cv2.imwrite(path, array)

                # Save dynamic masks
                for cam_id in range(len(camera_transforms)):
                    cv2.imwrite(
                        os.path.join(DYNAMIC_MASK_DIR, f"{frame_key}_{cam_id}.png"),
                        dyn_masks[cam_id],
                    )

                # Save sky masks from semantic cameras
                for cam_id, sem_img in sem_buffers.items():
                    if sem_img is None:
                        continue
                    arr = np.frombuffer(sem_img.raw_data, dtype=np.uint8) \
                            .reshape((sem_img.height, sem_img.width, 4))
                    sky_mask = (arr[:, :, 2] == SKY_LABEL).astype(np.uint8) * 255
                    cv2.imwrite(
                        os.path.join(SKY_MASK_DIR, f"{frame_key}_{cam_id}.png"),
                        sky_mask,
                    )

                # Save surface normal maps
                # CARLA outputs BGRA; convert BGR->RGB so PIL loads correctly in DeSiRe-GS
                from PIL import Image as _PIL
                for cam_id, norm_img in norm_buffers.items():
                    if norm_img is None:
                        continue
                    arr = np.frombuffer(norm_img.raw_data, dtype=np.uint8) \
                            .reshape((norm_img.height, norm_img.width, 4))
                    normal_rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGR -> RGB
                    _PIL.fromarray(normal_rgb).save(
                        os.path.join(NORMAL_DIR, f"{frame_key}_{cam_id}.jpg"),
                        quality=95,
                    )

                save_idx += 1

            frame += 1


    finally:
        with open(os.path.join(OUTPUT_DIR, "timestamps.json"), "w") as f:
            json.dump(timestamps, f, indent=1)

        # EmerNeRF frame_info.json — descriptive metadata (not used by training code)
        weather = world.get_weather()
        if weather.sun_altitude_angle > 0:
            time_of_day = "Day"
        elif weather.sun_altitude_angle > -6:
            time_of_day = "Dawn/Dusk"
        else:
            time_of_day = "Night"
        if weather.precipitation > 50:
            weather_str = "rain"
        elif weather.cloudiness > 60:
            weather_str = "cloudy"
        else:
            weather_str = "sunny"
        frame_info = {
            "time_of_day": time_of_day,
            "location": world.get_map().name.split("/")[-1],  # e.g. "Town01"
            "weather": weather_str,
            "TYPE_VEHICLE": len(vehicles),
        }
        with open(os.path.join(OUTPUT_DIR, "frame_info.json"), "w") as f:
            json.dump(frame_info, f, indent=1)

        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "pointcloud.npz"),
            pointcloud=pts_3d_all,
            camera_projection=pts_2d_all,
        )

        # Filter: keep only tracks visible in >= MIN_TRACK_FRAMES frames in at least one camera
        valid_track_ids = {
            tid for tid, lines in track_lines_by_id.items()
            if sum(
                1 for f, _ in lines
                if len(bbox_visible_dict.get(str(tid), {}).get(str(f), [])) > 0
            ) >= MIN_TRACK_FRAMES
        }
        bbox_visible_dict = {k: v for k, v in bbox_visible_dict.items()
                             if int(k) in valid_track_ids}
        all_lines = sorted(
            [(f, line) for tid in valid_track_ids for f, line in track_lines_by_id[tid]],
            key=lambda x: x[0],
        )
        with open(os.path.join(TRACK_DIR, "track_info.txt"), "w") as track_info_file:
            track_info_file.write(
                "frame_id track_id object_class alpha "
                "box_height box_width box_length "
                "box_center_x box_center_y box_center_z "
                "box_heading speed\n"
            )
            for _, line in all_lines:
                track_info_file.write(line)

        with open(os.path.join(TRACK_DIR, "track_camera_vis.json"), "w") as f:
            json.dump(bbox_visible_dict, f, indent=1)
        with open(os.path.join(TRACK_DIR, "track_ids.json"), "w") as f:
            json.dump(object_ids, f, indent=2)
        if track_vis_imgs:
            imageio.mimwrite(
                os.path.join(TRACK_DIR, "track_vis.mp4"),
                track_vis_imgs,
                fps=int(1 / settings.fixed_delta_seconds),
            )

        print("Cleaning up...")
        
        for cam in cameras:
            cam.stop()
            cam.destroy()
        for sem_cam in sem_cameras:
            sem_cam.stop()
            sem_cam.destroy()
        lidar.stop()
        lidar.destroy()
        ego_vehicle.destroy()
        for v in vehicles:
            v.destroy()

        settings.synchronous_mode = False
        world.apply_settings(settings)
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
