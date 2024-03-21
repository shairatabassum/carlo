import os
from pathlib import Path
from typing import Literal

import math
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
from src.experiments.experiment_settings import Experiment, GaussianNoise
from src.util.confirm_overwrite import confirm_path_overwrite
from src.util.create_camera_rigs_from_rig import create_camera_rigs_from_rig
from src.util.create_slurm_script import create_slurm_script
from src.util.carla_to_nerf import carla_to_marsnerf
from src.util.timer import Timer
from src.util.transform_file_mars import TransformFile
from examples.client_bounding_boxes import ClientSideBoundingBoxes


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

def apply_noise(transform: carla.Transform, noise: GaussianNoise):
    print(f"Applying noise to transform: {transform.location}")
    transform.location.x += np.random.normal(noise.mean, noise.std)
    transform.location.y += np.random.normal(noise.mean, noise.std)
    transform.location.z += np.random.normal(noise.mean, noise.std)

    print(f"Applied noise to transform: {transform.location}")
    return transform


def carla2Nerf(transform):
    #mat = np.array(transform.get_matrix())
    mat = transform
    rotz = np.array([[0.0000000,  -1.0000000,  0.0000000, 0.0],
                     [1.0000000,  0.0000000,  0.0000000, 0.0],
                     [0.0000000, 0.0000000,  1.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around z-axis
    roty = np.array([[0.0000000,  0.0000000,  1.0000000, 0.0],
                     [0.0000000,  1.0000000,  0.0000000, 0.0],
                     [-1.0000000,  0.0000000,  0.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around y-axis

    trafo1 = np.array([[0.0000000, 1.0000000, 0.0000000, 0.0],
                       [0.0000000, 0.0000000, 1.0000000, 0.0],
                       [-1.0000000, 0.0000000, 0.0000000, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    trafo2 = np.array([[0.0000000, 0.0000000, -1.0000000, 0.0],
                       [1.0000000, 0.0000000, 0.0000000, 0.0],
                       [0.0000000, 1.0000000, 0.0000000, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    carla2opengl = np.matmul(roty, rotz)
    #pose = np.matmul(mat, carla2opengl)
    #pose[0, 3] = -pose[0, 3]
    pose = np.matmul(trafo1, mat)
    pose = np.matmul(pose, trafo2)
    return pose


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
        print(f"✅ Saved experiment settings to {experiment_path / 'experiment_settings.txt'}")
    
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    time_dict = {}


    with Session(dt=0.1, phys_dt=0.01, phys_substeps=10) as session:

        # Run all the experiments in the same session.
        for index, run in enumerate(experiment.experiments):            
            ego = spawn_ego(autopilot=True, spawn_point=run.spawn_transform, filter="vehicle.tesla.model3")
            setup_traffic_manager(session.traffic_manager, ego, run.turns, run.percentage_speed_difference, run.path)
            vehicle_info = spawn_vehicles(count=25, autopilot=True, filter="vehicle")

            session.world.tick()
            w_frame = session.world.get_snapshot().frame

            # vehicles = []
            # vehicle_ids = [Actor.id for Actor in session.world.get_actors().filter('*vehicle*')]
            # print(vehicle_ids)
            # for x in range(len(vehicle_ids)):
            #     vehicles.append(session.world.get_actors().find(vehicle_ids[x]))
            # print(vehicles[0].get_transform())
            # print(vehicles[0].get_transform().get_matrix())
            # print(vehicles[0].get_transform().get_inverse_matrix())

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


            while not (should_stop(next_action, stop_next_straight, distance_traveled, run.stop_distance)):
                # timer_iter.tick('dt: {dt:.3f} s, avg: {avg:.3f} s, FPS: {fps:.1f} Hz')
                session.world.tick()
                # img = camera_rigs[0].get_image()
                # img = np.reshape(np.copy(img.raw_data), (img.height, img.width, 4))

                # Stack images together horizontally
                image = cv2.hconcat([camera_rig.get_image() for camera_rig in camera_rigs])
                cv2.imshow(window_title, image)
                image = None

                # Store image and update distance traveled every n-th tick.
                if image_tick % ticks_per_image == 0:
                    camera_rgb_ID = 0
                    camera_depth_ID = 0
                    for camera_rig in camera_rigs:
                        transform = camera_rig.camera.actor.get_transform()
                        transform = transform if run.location_noise is None else apply_noise(
                            transform, run.location_noise)
                        img = camera_rig.previous_image.astype(np.uint8)
                        if camera_rig.camtype=="rgb":
                            world_2_camera = transform.get_inverse_matrix()
                            veh_dict = {}
                            # Getting bounding box information
                            # bboxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, camera_rig.camera.actor, K)
                            
                            for npc in session.world.get_actors().filter('*vehicle*'):
                                if npc.id != ego.id:
                                    bb = npc.bounding_box
                                    speed = np.sum([np.abs(npc.get_velocity().x),np.abs(npc.get_velocity().y), np.abs(npc.get_velocity().z)])
                                    dist = npc.get_transform().location.distance(ego.get_transform().location)
                                    
                                    if speed > 1.0 and dist < 75:
                                        cam_dict = {}
                                        visible = False
                                        # Calculate the dot product between the forward vector of the vehicle and the vector between the vehicle
                                        # and the other vehicle. We threshold this dot product to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                                        forward_vec = transform.get_forward_vector()
                                        ray = npc.get_transform().location - ego.get_transform().location

                                        #if left_vec.dot(ray) > 1 or forward_vec.dot(ray) > 1 or right_vec.dot(ray) > 1:
                                        if forward_vec.dot(ray) > 1:
                                            box_dict = {}
                                            #print(speed)

                                            # PROCESS 1: take bounding box location, transform it to vehicle position
                                            # but it will give center point location and will require z-shifting
                                            # bbox_center = np.array(carla.Transform(bb.location, bb.rotation).get_matrix())
                                            # npc2w = np.array(npc.get_transform().get_matrix())
                                            # bbox_center2w = np.dot((npc2w), bbox_center)
                                            # w_x = bbox_center2w[0][3]
                                            # w_y = bbox_center2w[1][3]
                                            # w_z = bbox_center2w[2][3]
                                            # bbox_location = " ".join(map(str, [(w_x), (w_y), (w_z)]))
                                            
                                            # PROCESS 2: directly take the vehicle location, requires z-shifting to reach mid-point
                                            # npc_location = npc.get_transform().location
                                            # w_x = npc_location.x
                                            # w_y = npc_location.y
                                            # w_z = npc_location.z
                                            # # w_z += bb.extent.z #TO REACH MIDPOINT


                                            # PROCESS 3: Bounding box transform to Sensor/NPC (OLE FUNCTION)
                                            # npc_location = np.array(carla.Transform(bb.location, bb.rotation).get_matrix())
                                            # npc_location = np.array(npc.get_transform().get_matrix())
                                            # npc_location[2][3] -= bb.extent.z
                                            # npc2w = np.array(transform.get_matrix())
                                            # npc_world = np.dot(npc2w, npc_location)

                                            # npc_tran, npc_rot = mat2transform(npc_world)
                                            # npc_world_transform = carla.Transform(
                                            #     carla.Location(x=npc_tran[0], y=npc_tran[1], z=npc_tran[2]),
                                            #     carla.Rotation(pitch=npc_rot[0], yaw=npc_rot[1], roll=npc_rot[2])
                                            # )

                                            # npc_location = carla.Transform(bb.location, bb.rotation)
                                            # cam_matrix = transform.get_matrix()
                                            # proj = np.dot(cam_matrix, npc_matrix)
                                            # npc_tran, npc_rot = mat2transform(proj)
                                            # npc_world_transform = carla.Transform(
                                            #     carla.Location(x=npc_tran[0], y=npc_tran[1], z=npc_tran[2]),
                                            #     carla.Rotation(pitch=npc_rot[0], yaw=npc_rot[1], roll=npc_rot[2])
                                            # )
                                            # npc_location = np.array(npc.get_transform().get_matrix())
                                            # cam_transform = np.array(transform.get_matrix())
                                            # npc_to_camworld = np.dot(cam_transform, npc_location)
                                            # trans, rot = mat2transform(npc_to_camworld)
                                            # npc_transform = carla.Transform(
                                            #     carla.Location(x=trans[0], y=trans[1], z=trans[2]),
                                            #     carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
                                            # )
                                            # npc_world_transform_opengl = carla_to_marsnerf(npc_transform)
                                            npc_world_transform_opengl = carla_to_marsnerf(npc.get_transform())
                                            # npc_world_transform_opengl = np.array(npc_world_transform_opengl.get_matrix())
                                            
                                            # R_opengl = npc_world_transform_opengl[:3, :3]
                                            # t_opengl = npc_world_transform_opengl[:3, 3]
                                            # R_opencv = np.transpose(R_opengl)
                                            # R_opencv[1:3, :] *= -1
                                            # t_opencv = -np.dot(R_opencv, t_opengl)
                                            # pose_opencv = np.zeros((4, 4))
                                            # pose_opencv[:3, :3] = R_opencv
                                            # pose_opencv[:3, 3] = t_opencv
                                            # pose_opencv[3, 3] = 1
                                            
                                            # trans, rot = mat2transform(pose_opencv)
                                            # npc_world_transform_opengl = carla.Transform(
                                            #     carla.Location(x=trans[0], y=trans[1], z=trans[2]),
                                            #     carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
                                            # )
                                            
                                            w_x = npc_world_transform_opengl.location.x
                                            w_y = npc_world_transform_opengl.location.y
                                            w_z = npc_world_transform_opengl.location.z
                                            w_z -= bb.extent.z
                                            w_pitch = npc_world_transform_opengl.rotation.pitch * (np.pi/180)
                                            w_yaw = npc_world_transform_opengl.rotation.yaw * (np.pi/180)
                                            w_roll = npc_world_transform_opengl.rotation.roll * (np.pi/180)
                                            # w_pitch = npc_world_transform_opengl.rotation.pitch
                                            # w_yaw = npc_world_transform_opengl.rotation.yaw
                                            # w_roll = npc_world_transform_opengl.rotation.roll


                                            # define here the final value of obj_location and rotation
                                            obj_location = " ".join(map(str, [w_x, w_y, w_z]))
                                            obj_rotation = " ".join(map(str, [w_pitch, w_yaw, w_roll]))

                                            
                                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                            x_max = -10000
                                            x_min = 10000
                                            y_max = -10000
                                            y_min = 10000

                                            for vert in verts:
                                                p = get_image_point(vert, K, world_2_camera)
                                                # Find the rightmost vertex
                                                if p[0] > x_max:
                                                    x_max = p[0]
                                                # Find the leftmost vertex
                                                if p[0] < x_min:
                                                    x_min = p[0]
                                                # Find the highest vertex
                                                if p[1] > y_max:
                                                    y_max = p[1]
                                                # Find the lowest  vertex
                                                if p[1] < y_min:
                                                    y_min = p[1]
                                            if x_min < 0 and not x_max < 0 and x_max < image_w:
                                                x_min = 0
                                            if x_max > image_w and not x_min > image_w and x_min >= 0:
                                                x_max = image_w
                                            if y_min < 0 and not y_max < 0 and y_max < image_h:
                                                y_min = 0
                                            if y_max > image_h and not y_min > image_h and y_min >= 0:
                                                y_max = image_h
                                            if x_min >= 0 and x_max <= image_w and y_min >= 0 and y_max <= image_h:
                                                visible = True
                                                box_dict['world_location'] = obj_location
                                                box_dict['world_rotation'] = obj_rotation
                                                box_dict['extent'] = np.array([bb.extent.y, bb.extent.z, bb.extent.x])

                                                box_dict['x_max'] = int(x_max)
                                                box_dict['x_min'] = int(x_min)
                                                box_dict['y_max'] = int(y_max)
                                                box_dict['y_min'] = int(y_min)
                                                
                                                for edge in edges:
                                                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                                                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                                                veh_dict[str(npc.id)] = box_dict
                                                
                                                # rotational y
                                                # degree = ego.get_transform().rotation.yaw - npc.get_transform().rotation.yaw
                                                # deg_to_rad = degree * math.pi / 180
                                                # rotation_y = deg_to_rad % math.pi

                                                transform_file.append_bboxes(frameID, camera_rgb_ID, npc.id, int(x_min), int(x_max), int(y_min), int(y_max))
                                                transform_file.append_poses(frameID, camera_rgb_ID, npc.id, box_dict['extent'], box_dict['world_location'], box_dict['world_rotation'])
                                                transform_file.append_obj_extrinsics(frameID, camera_rgb_ID, npc.id, npc.get_transform())
                                            else:
                                                transform_file.append_bboxes(frameID, camera_rgb_ID, npc.id, 0, 0, 0, 0)
                                                transform_file.append_poses(frameID, camera_rgb_ID, npc.id, np.array([0, 0, 0]), "0 0 0", "0 0 0")
                                        else:
                                            transform_file.append_bboxes(frameID, camera_rgb_ID, npc.id, 0, 0, 0, 0)
                                            transform_file.append_poses(frameID, camera_rgb_ID, npc.id, np.array([0, 0, 0]), "0 0 0", "0 0 0")


                            time_dict[str(w_frame)] = veh_dict
                            # print(time_dict[str(w_frame)])
                            if image is None:
                                image = img
                            else:
                                image = cv2.hconcat([image, img])
                            transform_file.append_frame(camera_rig.previous_image, transform, camera_rig.camtype, camera_rgb_ID, frameID)
                            camera_rgb_ID += 1
                        elif camera_rig.camtype=="depth":
                            transform_file.append_frame(camera_rig.previous_image, transform, camera_rig.camtype, camera_depth_ID, frameID)
                            camera_depth_ID += 1
                            image = cv2.hconcat([image, img])

                    cv2.imshow(window_title, image)

                    
                    current_location = ego.get_location()
                    distance_traveled += get_distance_traveled(prev_location, current_location)
                    prev_location = current_location
                    print(f"Total distance traveled: {distance_traveled:.2f} meters")
                    frameID += 1

                # Determine if we should stop the next straight
                next_action = session.traffic_manager.get_next_action(ego)[0]
                # if (next_action == "Left" and previous_action != "Left"):
                #     turns += 1
                #     if (turns == run.turns):
                #         stop_next_straight = True
                if distance_traveled>=80:
                    stop_next_straight = True
                previous_action = next_action

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                image_tick += 1

            transform_file.export_transforms()
            transform_file.export_bbox()
            transform_file.export_pose()
            transform_file.export_vehicle_info(vehicle_info=vehicle_info)
            
            destroy_actors(session.world, "vehicle*")
            destroy_actors(session.world, "sensor*")
            print("\n\nNEXT EXPERIMENT\n\n")

        cv2.destroyWindow(window_title)


experiment = experiments.experiment_test
run_session(experiment)
