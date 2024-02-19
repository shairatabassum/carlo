from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import carla
import numpy as np

from .sensor import SensorBase


@dataclass
class CameraSettings:
    fov: Optional[float] = None
    fstop: Optional[float] = None
    image_size_x: Optional[int] = None
    image_size_y: Optional[int] = None
    iso: Optional[float] = None
    gamma: Optional[float] = None
    shutter_speed: Optional[float] = None


class Camera(SensorBase[carla.Image, CameraSettings]):
    DEFAULT_BLUEPRINT = 'sensor.camera.rgb'

    #init method from SensorBase to get all information
    def __init__(self, *,
                 actor: Optional[carla.Sensor] = None,
                 settings: Union[None, CameraSettings, Dict[str, Any]] = None,
                 blueprint: Union[None, str, carla.ActorBlueprint] = None,
                 transform: Optional[carla.Transform] = None,
                 parent: Optional[carla.Actor] = None,
                 attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
                 ) -> None:
        super().__init__(actor=actor, settings=settings, blueprint=blueprint,
                         transform=transform, parent=parent,
                         attachment_type=attachment_type)
        self.blueprint = blueprint
        
    
    def add_numpy_queue(self) -> 'Queue[np.ndarray]':
        """Creates a queue that receives camera images as numpy arrays."""

        def transform(image: carla.Image) -> np.ndarray:
            #converting original to logarithmic depth
            if self.blueprint == 'sensor.camera.depth':
                image.convert(carla.ColorConverter.LogarithmicDepth)
            data = np.frombuffer(image.raw_data, dtype=np.uint8)
            data = np.reshape(data, (image.height, image.width, 4))
            return data

        return self.add_queue(transform=transform)
    
    def __str__(self) -> str:
        return super().__str__() + f", settings={self.settings}"
