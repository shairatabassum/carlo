from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional
import carla

from src.sensors.camera_rig import CameraRig

@dataclass
class GaussianNoise:
    mean: float = 0.0
    std: float = 1.0

@dataclass
class ExperimentSettings:
    camera_rigs: Optional[List[CameraRig]] = None
    rig_file_path: Optional[str] = None
    ticks_per_image: int = 3
    turns: int = 3 # Equivalent to one lap around the block
    stop_distance: Optional[int] = None
    percentage_speed_difference: int = 0
    location_noise: Optional[GaussianNoise] = None
    spawn_transform: carla.Transform = carla.Transform(carla.Location(x=106.386559, y=-2.362594, z=0.5),
                                                       carla.Rotation(pitch=0, yaw=-90, roll=0))
    # path: Literal["left-loop", "city-wander"] = "left-loop"
    path: Literal["left-loop", "city-wander"] = "city-wander"

@dataclass
class Experiment:
    experiments: List[ExperimentSettings]
    experiment_name: str = field(default_factory=lambda: str(int(datetime.timestamp(datetime.now()))))