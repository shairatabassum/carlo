import os
from pathlib import Path
from typing import Literal

import open3d as o3d
import numpy as np
from math import atan2, asin, sqrt, degrees
from numpy.linalg import pinv, inv
import queue
import cv2
import carla
from src.common.rig import parse_rig_json

from src.common.session import Session
from src.common.spawn import spawn_ego, spawn_vehicles
from src.experiments import experiments
from src.sensors.lidar import Lidar, LidarSettings
from src.experiments.experiment_settings import Experiment, GaussianNoise
from src.util.confirm_overwrite import confirm_path_overwrite
from src.util.create_camera_rigs_from_rig import create_camera_rigs_from_rig
from src.util.carla_to_nerf import carla_to_marsnerf
from src.util.timer import Timer
from src.util.transform_file_mars import TransformFile
from examples.client_bounding_boxes import ClientSideBoundingBoxes

#lidar parameters
freq = 10
lidar_channels = 64
lidar_horizontal_resolution = 1  # degrees

def mat2transform(M):
    R = M[:3, :3]
    T = M[:3, 3]
    yaw = atan2(R[1, 0], R[0, 0])
    pitch = atan2(-R[2, 0], sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = atan2(R[2, 1], R[2, 2])

    determinant = np.linalg.det(R)
    if determinant > 0:  # Left-handed system
        pitch = -pitch
        roll = -roll
    return T, (degrees(pitch), degrees(yaw), degrees(roll))


def setup_traffic_manager(traffic_manager: carla.TrafficManager, ego: carla.Actor, turns: int, percentage_speed_difference: int, path: Literal["left-loop", "city-wander"]):
    traffic_manager.ignore_lights_percentage(ego, 100)  # Ignore traffic lights 100% of the time
    traffic_manager.vehicle_percentage_speed_difference(
        ego, percentage_speed_difference)  # 100% slower than speed limit
    if path == "left-loop":
        traffic_manager.set_route(ego, ["Left"] * turns)
    elif path == "city-wander":
        # TODO: Make this deterministic
        pass # Don't specify a route


def get_distance_traveled(prev_location, current_location):
    return np.sqrt((current_location.x - prev_location.x)**2 + (current_location.y - prev_location.y)**2 + (current_location.z - prev_location.z)**2)


# Stops if number of turns is reached or if distance traveled is greater than stop_distance
def should_stop(next_action, stop_next_straight, distance_traveled, stop_distance):
    if next_action == "LaneFollow" and stop_next_straight:
        return True

    if stop_distance is not None and distance_traveled >= stop_distance:
        return True

    return False

def destroy_actors(world: carla.World, actor_filter: str):
    actor_list = world.get_actors().filter(actor_filter)
    for actor in actor_list:
        if actor.is_alive:
            actor.destroy()


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def run_session(experiment: Experiment):

    # Create directory for experiment
    root_path = Path(os.curdir)
    experiment_path = root_path / "runs_mars" / experiment.experiment_name
    os.makedirs(experiment_path, exist_ok=True)

    # Save the experiment settings to the experiment directory
    settings_path = experiment_path / "experiment_settings.txt"
    confirm_path_overwrite(settings_path)
    with open(settings_path, "w") as f:
        f.write(str(experiment))
        print(f"Saved experiment settings to {experiment_path / 'experiment_settings.txt'}")
    
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    time_dict = {}


    with Session(dt=0.1, phys_dt=0.01, phys_substeps=10) as session:

        # Run all the experiments in the same session.
        for index, run in enumerate(experiment.experiments):            
            ego = spawn_ego(autopilot=True, spawn_point=run.spawn_transform, filter="vehicle.tesla.model3")
            setup_traffic_manager(session.traffic_manager, ego, run.turns, run.percentage_speed_difference, run.path)
            vehicle_info = spawn_vehicles(count=0, autopilot=True, filter="vehicle.*")
            
            lidar = Lidar(parent=ego,
                  transform=carla.Transform(carla.Location(z=3.0)),
                  settings=LidarSettings(
                      range=100,
                      noise_stddev=0.1,
                      upper_fov=25,
                      lower_fov=-25,
                      channels=lidar_channels,
                      rotation_frequency=freq,
                      points_per_second=lidar_channels * freq * round(360 / lidar_horizontal_resolution),
                  ))
            lidar_queue = lidar.add_pointcloud_queue()
            lidar_np_queue = lidar.add_numpy_queue()
            lidar.start()
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name='LiDAR',
                width=960,
                height=540,
                left=100,
                top=100
            )

            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 1.0

            point_cloud = o3d.geometry.PointCloud()
            added = False

            session.world.tick()
            w_frame = session.world.get_snapshot().frame

            image_tick = 0
            ticks_per_image = run.ticks_per_image
            previous_action = None
            turns = 0
            stop_next_straight = False
            next_action = None
            distance_traveled = 0
            prev_location = run.spawn_transform.location
            frameID = 0

            # Create cameras
            camera_rigs = [camera_rig.create_camera(ego) for camera_rig in run.camera_rigs] if run.camera_rigs is not None else []
            if run.rig_file_path is not None:
                rig = rig = parse_rig_json(run.rig_file_path)
                camera_rigs = create_camera_rigs_from_rig(ego=ego, rig=rig)

            timer_iter = Timer()
            window_title = 'Camera'
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            # Create a TransformFile
            transform_file = TransformFile(output_dir=experiment_path, camera_rigs=camera_rigs)

            # Set the intrinsics of the camera
            camera_settings = camera_rigs[0].get_camera_settings()
            image_w = camera_settings.image_size_x
            image_h = camera_settings.image_size_y
            K = transform_file.get_intrinsics(image_w, image_h, camera_settings.fov)

            #lidar parameters
            pts_3d_all = {}
            pts_2d_all = {}
            while not (should_stop(next_action, stop_next_straight, distance_traveled, run.stop_distance)):
                session.world.tick()

                # Stack images together horizontally
                image = cv2.hconcat([camera_rig.get_image() for camera_rig in camera_rigs])
                cv2.imshow(window_title, image)
                image = None

                # Store image and update distance traveled every n-th tick.
                if image_tick % ticks_per_image == 0:
                    
                    try:
                        lidar_data = lidar_np_queue.get(timeout=1.0)
                        lidar_data_vis = lidar_queue.get(timeout=1.0)
                    except queue.Empty:
                        print("No LiDAR data")
                        # lidar_data = None
                    
                    point_cloud.points = lidar_data_vis.points
                    point_cloud.colors = lidar_data_vis.colors

                    if not added:
                        vis.add_geometry(point_cloud)
                        added = True

                    vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()
                    
                    
                    if lidar_data is not None and len(lidar_data) > 0:
                        points = lidar_data[:, :3]
                        points_converted = np.stack([
                        points[:, 1],     # right
                        -points[:, 2],    # down
                        points[:, 0]      # forward
                        ], axis=1)
                        pts_3d_all[frameID] = points_converted
                    
                    camera_rgb_ID = 0
                    projections = []
                    for cam_id, camera_rig in enumerate(camera_rigs):
                        transform = camera_rig.camera.actor.get_transform()
                        w2c = np.array(transform.get_inverse_matrix())
                        
                        # project all points for lidar
                        v2w = np.array(ego.get_transform().get_matrix())
                        pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
                        pts_world = (v2w @ pts_h.T).T[:, :3]
                        pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
                        pts_cam = (w2c @ pts_h.T).T
                        # convert to standard camera coords
                        pts_cam = np.stack([
                            pts_cam[:, 1],
                            -pts_cam[:, 2],
                            pts_cam[:, 0]
                        ], axis=1)
                        valid = pts_cam[:, 2] > 0
                        pts_proj = np.full((points.shape[0], 3), -1)
                        pts_proj[valid, 0] = (K[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2]) + K[0, 2]
                        pts_proj[valid, 1] = (K[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2]) + K[1, 2]
                        pts_proj[valid, 2] = cam_id
                        projections.append(pts_proj)
                        
                        img = camera_rig.previous_image.copy().astype(np.uint8)
                        for p in pts_proj:
                            x, y = int(p[0]), int(p[1])
                            if x >= 0 and y >= 0 and x < image_w and y < image_h:
                                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                        
                        img = camera_rig.previous_image.astype(np.uint8)
                        
                        if camera_rig.camtype=="rgb":
                            world_2_camera = transform.get_inverse_matrix()
                            veh_dict = {}
                            time_dict[str(w_frame)] = veh_dict
                            if image is None:
                                image = img
                            else:
                                image = cv2.hconcat([image, img])
                            transform_file.append_frame(camera_rig.previous_image, transform, camera_rig.camtype, camera_rgb_ID, frameID)
                            camera_rgb_ID += 1
                    projections = np.stack(projections, axis=1)  # (N, num_cam, 3)
                    final_proj = np.full((points.shape[0], 6), -1)
                    for i in range(points.shape[0]):
                        valid_cams = np.where(projections[i,:,0] != -1)[0]

                        if len(valid_cams) > 0:
                            c = valid_cams[0]
                            final_proj[i, 0:3] = projections[i, c]

                        if len(valid_cams) > 1:
                            c = valid_cams[1]
                            final_proj[i, 3:6] = projections[i, c]

                    pts_2d_all[frameID] = final_proj.astype(np.int16)
                    cv2.imshow(window_title, image)

                    
                    current_location = ego.get_location()
                    distance_traveled += get_distance_traveled(prev_location, current_location)
                    prev_location = current_location
                    print(f"Total distance traveled: {distance_traveled:.2f} meters")
                    frameID += 1

                # Determine if we should stop the next straight
                next_action = session.traffic_manager.get_next_action(ego)[0]

                if distance_traveled>=30000:
                    stop_next_straight = True
                previous_action = next_action

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                image_tick += 1
            
            np.savez_compressed(
                experiment_path / "pointcloud.npz",
                pointcloud=pts_3d_all,
                camera_projection=pts_2d_all
            )
            
            transform_file.export_transforms()
            transform_file.export_bbox()
            transform_file.export_pose()
            transform_file.export_vehicle_info(vehicle_info=vehicle_info)
            
            destroy_actors(session.world, "vehicle*")
            destroy_actors(session.world, "sensor*")
            print("\n\nNEXT EXPERIMENT\n\n")

        cv2.destroyWindow(window_title)


experiment = experiments.experiment_waymo
run_session(experiment)
