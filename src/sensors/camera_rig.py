import numpy as np
from src.sensors.camera import Camera, CameraSettings
import carla
from queue import Queue


class CameraRig:
    base_camera_settings = CameraSettings(image_size_x=600, image_size_y=450, fov=90)
    base_rotation = carla.Rotation(yaw=0, pitch=0, roll=0)
    base_location = carla.Location(z=3.0)
    base_transform = carla.Transform(base_location, base_rotation)

    def __init__(self, transform: carla.Transform = base_transform, camera_settings: CameraSettings = base_camera_settings, camtype: str = "rgb"):
        self.transform = transform
        self.camera_settings = camera_settings
        self.camtype = camtype
        self.camera = None
        self.camera_queue = None
        self.previous_image = None

    def create_camera(self, parent: carla.Actor) -> 'CameraRig':
        if self.camtype=='rgb':
            self.camera = Camera(parent=parent, transform=self.transform, settings=self.camera_settings)
        elif self.camtype=='depth':
            self.camera = Camera(parent=parent, blueprint='sensor.camera.depth', transform=self.transform, settings=self.camera_settings)
        elif self.camtype=='class_seg':
            self.camera = Camera(parent=parent, blueprint='sensor.camera.semantic_segmentation', transform=self.transform, settings=self.camera_settings)
        elif self.camtype=='inst_seg':
            self.camera = Camera(parent=parent, blueprint='sensor.camera.instance_segmentation', transform=self.transform, settings=self.camera_settings)
        self.camera_queue = self.camera.add_numpy_queue()
        self.camera.start()
        return self

    def get_camera_settings(self) -> CameraSettings:
        return self.camera_settings

    def get_camera(self) -> Camera:
        if self.camera is None:
            raise Exception("Camera not created yet")
        return self.camera
    
    def get_camera_queue(self) -> 'Queue[np.ndarray]':
        if self.camera_queue is None:
            raise Exception("Camera not created yet")
        return self.camera_queue

    def get_image(self) -> np.ndarray:
        image = self.camera_queue.get()
        self.previous_image = image
        return image

    def __str__(self) -> str:
        return f"CameraRig(transform={self.transform}, camera_settings={self.camera_settings})"
    
    def __repr__(self) -> str:
        return self.__str__()
