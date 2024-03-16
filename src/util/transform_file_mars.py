import json
import math
import os
from pathlib import Path
import cv2
import carla
import numpy as np

from datetime import datetime

from src.util.carla_to_nerf import carla_to_nerf


class TransformFile:
    def __init__(self, output_dir=None, camera_rigs=None) -> None:
        self.frames = []
        self.intrinsics = []
        self.extrinsics = []
        self.intrinsic_matrix = {}
        self.bboxes = []
        self.poses = []
        self.info = []

        root_path = Path(os.curdir)
        self.output_dir = root_path / output_dir

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.image_dir = self.output_dir / 'frames' / 'rgb'
        self.depth_dir = self.output_dir / 'frames' / 'depth'
        self.image_dir.mkdir(exist_ok=True, parents=True)
        self.depth_dir.mkdir(exist_ok=True, parents=True)

        camera_rgb_ID = 0
        camera_depth_ID = 0
        for camera_rig in camera_rigs:
            if camera_rig.camtype == "rgb":
                cam_folder = os.path.join(self.image_dir, f'Camera_{camera_rgb_ID}')
                os.makedirs(cam_folder, exist_ok=True)
                camera_rgb_ID += 1
            elif camera_rig.camtype == "depth":
                cam_folder = os.path.join(self.depth_dir, f'Camera_{camera_depth_ID}')
                os.makedirs(cam_folder, exist_ok=True)
                camera_depth_ID += 1

    def append_frame(self, image: np.ndarray, transform: carla.Transform, camtype: str = "rgb", cameraID: int = 0, frameID: int = 0):
        # Save the image to output
        if camtype == "rgb":
            file_path = str(self.image_dir / f'Camera_{cameraID}' / f'rgb_{frameID:05d}.jpg')
            cv2.imwrite(file_path, image)

            transform = carla_to_nerf(transform)
            ext_transform = ' '.join(str(item) for sublist in transform for item in sublist)
            self.extrinsics.append(f"{frameID} {cameraID} {ext_transform}\n")

            transform_intrinsic = ' '.join(str(item) for sublist in self.intrinsic_matrix for item in sublist)
            self.intrinsics.append(f"{frameID} {cameraID} {transform_intrinsic}\n")
            
        elif camtype == "depth":
            file_path = str(self.depth_dir / f'Camera_{cameraID}' / f'depth_{frameID:05d}.png')
            cv2.imwrite(file_path, image)
            # self.countDepth += 1
        
    def append_bboxes(self, frameID, cameraID, trackID, x_min, x_max, y_min, y_max):
        self.bboxes.append(
            f"{frameID} {cameraID} {trackID} {x_min} {x_max} {y_min} {y_max} 0 0 0 True\n"
            )

    def append_poses(self, frameID, cameraID, trackID, dimension, location, rotation):
        width = "{:.6f}".format(2*dimension[0])
        height = "{:.6f}".format(2*dimension[1])
        length = "{:.6f}".format(2*dimension[2])
        obj_dimension = " ".join(map(str, [width, height, length]))
        # w_x, w_y, w_z = [float(x) for x in location][0:3]
        # w_x, w_y, w_z = [row[3] for row in location[:-1]]
        # w_x = "{:.9f}".format(location[0][0])
        # w_y = "{:.9f}".format(location[1][0])
        # w_z = "{:.9f}".format(location[2][0])
        # w_x = location.x
        # w_y = location.y
        # w_z = location.z
        # w_z -= dimension[1]
        # obj_location = " ".join(map(str, [float(w_y), -float(w_z), float(w_x)]))
        self.poses.append(
            f"{frameID} {cameraID} {trackID} 0 {obj_dimension} {location} {rotation} 0 0 0 0 0 0\n"
        )

    def get_intrinsics(self, image_size_x, image_size_y, fov):
        intrinsic_matrix = np.identity(3)
        intrinsic_matrix[0, 2] = image_size_x / 2.0
        intrinsic_matrix[1, 2] = image_size_y / 2.0
        intrinsic_matrix[0, 0] = intrinsic_matrix[1, 1] = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
        self.intrinsic_matrix = intrinsic_matrix              
        return self.intrinsic_matrix

    def export_transforms(self, file_path='transforms.json'):
        extrinsic_path = self.output_dir / 'extrinsic.txt'
        with open(extrinsic_path, 'w+') as f:
            f.write("frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1\n")
            for ext in self.extrinsics:
                f.write(ext)
        print(f"Saved extrinsics to {extrinsic_path}")
        intrinsic_path = self.output_dir / 'intrinsic.txt'
        with open(intrinsic_path, 'w+') as f:
            f.write("frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n")
            for intr in self.intrinsics:
                f.write(intr)
            print(f"Saved intrinsics to {intrinsic_path}")

    def export_bbox(self, file_path='bbox.txt'):
        output_path = self.output_dir / file_path
        with open(output_path, 'w+') as f:
            f.write("frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving\n")
            for bboxes in self.bboxes:
                f.write(bboxes)
        print(f"Saved bounding boxes to {output_path}")

    def export_pose(self, file_path='pose.txt'):
        output_path = self.output_dir / file_path
        with open(output_path, 'w+') as f:
            f.write("frame cameraID trackID alpha width height length world_space_X world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z\n")
            for poses in self.poses:
                f.write(poses)
        print(f"Saved vehicle poses to {output_path}")

    def export_vehicle_info(self, file_path='info.txt', vehicle_info: str = ""):
        # getting all visible vehicle ids
        unique_ids = set()
        for line in self.bboxes:
            unique_ids.add(line.split()[2])
        
        output_path = self.output_dir / file_path
        with open(output_path, 'w+') as f:
            f.write("trackID label model color\n")
            for line in vehicle_info:
                if line.split()[0] in unique_ids:
                    f.write(line)
        print(f"Saved vehicle infos to {output_path}")