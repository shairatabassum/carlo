from dataclasses import dataclass
from typing import List
from src.experiments.experiment_settings import Experiment, ExperimentSettings, GaussianNoise
from src.sensors.camera import CameraSettings
from src.sensors.camera_rig import CameraRig
import carla

overhead_camera_transform = carla.Transform(carla.Location(z=12.7), carla.Rotation(pitch=-90))
base_camera_rig = [
    CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
    CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=30))),
]

experiment_test = Experiment(
    experiment_name='exp_test',
    experiments=[
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=base_camera_rig
        ),
    ]
)

##############################################
# Experiment 1: Camera-setup
##############################################


experiment_1 = Experiment(
    experiment_name='exp_camera_setup',
    experiments=[
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-10))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=10))),
            ]
        ),
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=30))),
            ]
        ),
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-50))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=50))),
            ]
        ),
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-70))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=70))),
            ]
        ),
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-50))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=0))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=50))),
            ]
        ),
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-70))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=0))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=70))),
            ]
        ),
    ]
)

##############################################
# Experiment 2: Capacity - how many turns? how many meters?
##############################################

experiment_2 = Experiment(
    experiment_name='exp_capacity',
    experiments=[
        ExperimentSettings(
            stop_distance=50,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            turns=1,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            turns=2,
            camera_rigs=base_camera_rig
        ),
        # One lap
        ExperimentSettings(
            turns=3,
            camera_rigs=base_camera_rig
        ),
        # Two laps
        ExperimentSettings(
            turns=6,
            camera_rigs=base_camera_rig
        ),
    ],
)

##############################################
# Experiment 3: Camera settings - image size
##############################################

camera_settings = [
    CameraSettings(image_size_x=200, image_size_y=150, fov=90),
    CameraSettings(image_size_x=400, image_size_y=300, fov=90),
    # CameraSettings(image_size_x=1242, image_size_y=375, fov=90),
    CameraSettings(image_size_x=1920, image_size_y=1080, fov=90),
    CameraSettings(image_size_x=1200, image_size_y=900, fov=90),
    CameraSettings(image_size_x=1600, image_size_y=1200, fov=90),
    CameraSettings(image_size_x=3840, image_size_y=2160, fov=90),
]

experiment_3 = Experiment(
    experiment_name='exp_image_size',
    experiments=[
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-30)), camera_settings=camera_settings[0]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=30)), camera_settings=camera_settings[0]),
            ]
        ),
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-30)),  camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=30)),  camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-30)),  camera_settings=camera_settings[2]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=30)),  camera_settings=camera_settings[2]),
            ]
        ),
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-30)),  camera_settings=camera_settings[3]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=30)),  camera_settings=camera_settings[3]),
            ]
        ),
        ExperimentSettings(
            stop_distance=100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-30)),  camera_settings=camera_settings[4]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=30)),  camera_settings=camera_settings[4]),
            ]
        ),
    ]
)

##############################################
# Experiment 5: Can COLMAP handle a full lap? ⛔

# WITH VOCAB-TREE
# COLMAP only found poses for 4.37% of the images. This is low.
# This can be caused by a variety of reasons, such poor scene coverage, blurry images, or large exposure changes.

# WITH SEQUENTIAL MATCHING
# COLMAP found poses for all images, CONGRATS!

# "fl_x": 199.95200841392978,
# "fl_y": 180.0158372186924,
# "cx": 199.86080786048282,
# "cy": 150.04495224409698,
# "w": 400,
# "h": 300,
# "camera_model": "OPENCV",
# "k1": -0.00031343408304528145,
# "k2": -0.0005933858192653826,
# "p1": -0.00012972942324782972,
# "p2": -0.00021240782388856521,


# THE GENERATED TRANSFORMS.JSON'S INTERNAL PARAMETERS
# "fl_x": 123.47392475671101,
# "fl_y": 92.60544356753326,
# "cx": 200.0,
# "cy": 150.0,
# "w": 400,
# "h": 300,
# "k1": 0,
# "k2": 0,
# "p1": 0,
# "p2": 0,
##############################################

experiment_5 = Experiment(
    experiment_name="exp_full_lap",
    experiments=[
        ExperimentSettings(
            camera_rigs=base_camera_rig
        ),
    ],
)

##############################################
# Experiment 7: Change the speed of the car
##############################################

experiment_7 = Experiment(
    experiment_name="exp_speed",
    experiments=[
        # 50% below speed-limit
        ExperimentSettings(
            turns=1,
            percentage_speed_difference=50,
            camera_rigs=[
                    CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
                    CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=30))),
            ]
        ),
        # Speed-limit
        ExperimentSettings(
            turns=1,
            percentage_speed_difference=0,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=30))),
            ]
        ),
        # 50% above speed-limit
        ExperimentSettings(
            turns=1,
            percentage_speed_difference=-50,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=30))),
            ]
        ),
        # 100% above speed-limit
        ExperimentSettings(
            turns=1,
            percentage_speed_difference=-100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=-30))),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=30))),
            ]
        ),
    ],
)

##############################################
# Experiment 8: Number of frames
##############################################

experiment_8 = Experiment(
    experiment_name="exp_frames_5",
    experiments=[
        ExperimentSettings(
            ticks_per_image=1,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            ticks_per_image=2,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            ticks_per_image=3,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            ticks_per_image=4,
            camera_rigs=base_camera_rig
        ),
        ExperimentSettings(
            ticks_per_image=5,
            camera_rigs=base_camera_rig
        ),
    ]
)

##############################################
# Experiment 9 and 10: Combined baseline
##############################################

# experiment_test = Experiment(
#     experiment_name="three_cam_bikes",
#     experiments=[
#         ExperimentSettings(
#             ticks_per_image=2,
#             percentage_speed_difference=50,
#             turns=3,
#             camera_rigs=[
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=-30)), camera_settings=camera_settings[2], camtype='rgb'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=-30)), camera_settings=camera_settings[2], camtype='depth'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=-30)), camera_settings=camera_settings[2], camtype='class_seg'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=-30)), camera_settings=camera_settings[2], camtype='inst_seg'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='rgb'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='depth'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='class_seg'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='inst_seg'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=30)), camera_settings=camera_settings[2], camtype='rgb'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=30)), camera_settings=camera_settings[2], camtype='depth'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=30)), camera_settings=camera_settings[2], camtype='class_seg'),
#                 CameraRig(transform=carla.Transform(carla.Location(z=3.0),
#                                                     carla.Rotation(yaw=30)), camera_settings=camera_settings[2], camtype='inst_seg'),
#             ]
#         ),
#     ]
# )

experiment_test = Experiment(
    experiment_name="static_scene",
    experiments=[
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='rgb'),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='depth'),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=0)), camera_settings=camera_settings[2], camtype='inst_seg'),
            ]
        ),
    ]
)

experiment_9 = Experiment(
    experiment_name="exp_combined_baseline",
    experiments=[
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=1,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ]
)

experiment_10 = Experiment(
    experiment_name="exp_combined_baseline_2",
    experiments=[
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ]
)

# Based on best settings from previous experiments.
# Ticks per image is set to 3 instead of 2, in order to reduce the amount of images.
# The turns are set to 3 instead of 1, in order to complete a full lap
baseline_experiment_settings = ExperimentSettings(
    ticks_per_image=3,
    percentage_speed_difference=50,
    turns=3,
    camera_rigs=[
        CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                            carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
        CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                            carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
    ]
),

##############################################
# Experiment 11: Gaussian noise to car location. How much Gaussian noise can camera optimization handle?
##############################################

experiment_11 = Experiment(
    experiment_name="exp_gaussian_noise_3",
    experiments=[
        # ExperimentSettings(
        #     ticks_per_image=3,
        #     percentage_speed_difference=50,
        #     turns=3,
        #     location_noise=None,
        #     camera_rigs=[
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
        #     ]
        # ),
        # ExperimentSettings(
        #     ticks_per_image=3,
        #     percentage_speed_difference=50,
        #     turns=3,
        #     location_noise=GaussianNoise(mean=0, std=0.5),  # 0.5 meters
        #     camera_rigs=[
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
        #     ]
        # ),
        # ExperimentSettings(
        #     ticks_per_image=3,
        #     percentage_speed_difference=50,
        #     turns=3,
        #     location_noise=GaussianNoise(mean=0, std=1),  # 1 meters
        #     camera_rigs=[
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
        #     ]
        # ),
        # ExperimentSettings(
        #     ticks_per_image=3,
        #     percentage_speed_difference=50,
        #     turns=3,
        #     location_noise=GaussianNoise(mean=0, std=3),  # 3 meters
        #     camera_rigs=[
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
        #         CameraRig(transform=carla.Transform(carla.Location(z=3.0),
        #                                             carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
        #     ]
        # ),
    ]
)

##############################################
# Experiment 12: Camera setup #2
##############################################

experiment_12 = Experiment(
    experiment_name='exp_camera_setup_straight',
    experiments=[
        ExperimentSettings(
            stop_distance=125,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0), carla.Rotation(yaw=0))),
            ]
        ),
    ]
)

##############################################
# Experiment 13: Camera optimization - less noise
##############################################

experiment_13 = Experiment(
    experiment_name="exp_gaussian_noise_2",
    experiments=[
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            turns=3,
            location_noise=None,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.1),  # 0.5 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.2),  # 1 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.3),  # 3 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ]
)

##############################################
# Experiment 14: Camera optimization - less noise, shorter segment
##############################################

experiment_14 = Experiment(
    experiment_name="exp_gaussian_noise_shorter_segments",
    experiments=[
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            stop_distance=50,
            location_noise=None,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            stop_distance=50,
            location_noise=GaussianNoise(mean=0, std=0.1),  # 0.5 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            stop_distance=50,
            location_noise=GaussianNoise(mean=0, std=0.2),  # 1 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            stop_distance=50,
            location_noise=GaussianNoise(mean=0, std=0.3),  # 3 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ]
)

##############################################
# Experiment 15: Block-NeRF: Long path
##############################################

experiment_15 = Experiment(
    experiment_name="exp_block_nerf_long_path",
    experiments=[
            ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            stop_distance=1200,
            location_noise=None,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ],
            path="city-wander"
        ),
    ]
)

##############################################
# Experiment 16: TEST
##############################################

experiment_16 = Experiment(
    experiment_name="test",
    experiments=[
            ExperimentSettings(
            ticks_per_image=3,
            percentage_speed_difference=50,
            stop_distance=50,
            location_noise=None,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[3]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[3]),
            ],
            path="city-wander"
        ),
    ]
)

##############################################
# Experiment 17: Run with rig_file
##############################################

experiment_17 = Experiment(
    experiment_name="exp_rig_file",
    experiments=[
            ExperimentSettings(
                ticks_per_image=2,
                percentage_speed_difference=50,
                stop_distance=50,
                location_noise=None,
                camera_rigs=None,
                rig_file_path="rig_files/rig2023.json",
                path="left-loop"
            ),
            ExperimentSettings(
                ticks_per_image=2,
                percentage_speed_difference=50,
                turns=3,
                location_noise=None,
                camera_rigs=None,
                rig_file_path="rig_files/rig2023.json",
                path="left-loop"
            ),
    ]
)

##############################################
# Experiment 18: One lap baseline, with fewer images
##############################################

experiment_18 = Experiment(
    experiment_name="exp_one_lap_baseline_fewer_images",
    experiments=[
        ExperimentSettings(
            ticks_per_image=4,
            percentage_speed_difference=10,
            turns=3,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ])


##############################################
# Experiment 19: Run to get images from the CARLA-simulator
##############################################

experiment_19 = Experiment(
    experiment_name="exp_get_images",
    experiments=[
        ExperimentSettings(
            ticks_per_image=4,
            percentage_speed_difference=10,
            turns=3,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-50)), camera_settings=camera_settings[2]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=0)), camera_settings=camera_settings[2]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=50)), camera_settings=camera_settings[2]),
            ]
        ),
    ])

##############################################
# Experiment 20: Test change in rotation
##############################################

experiment_20 = Experiment(
    experiment_name="exp_change_in_rotation",
    experiments=[
        ExperimentSettings(
            ticks_per_image=4,
            percentage_speed_difference=10,
            stop_distance=30,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ])

##############################################
# Experiment 21: Capacity - The last one; 2 laps
##############################################

experiment_21 = Experiment(
    experiment_name='exp_capacity_two_laps',
    experiments=[
        # Two laps
        ExperimentSettings(
            turns=6,
            camera_rigs=base_camera_rig
        ),
    ],
)

##############################################
# Experiment 22: Gaussian noise: REDONE
##############################################

experiment_22 = Experiment(
    experiment_name="exp_gaussian_noise_REDONE",
    experiments=[
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=None,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.1),  # 0.1 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.2),  # 0.2 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.3),  # 0.3 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=0.5),  # 0.5 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=1),  # 1 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        ExperimentSettings(
            ticks_per_image=2,
            percentage_speed_difference=50,
            turns=3,
            location_noise=GaussianNoise(mean=0, std=3),  # 3 meters
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ]
)

##############################################
# Experiment 23: Speed - REDONE
##############################################

experiment_23 = Experiment(
    experiment_name="exp_speed_REDONE_BASELINE",
    experiments=[
        # 50% below speed-limit
        ExperimentSettings(
            turns=3,
            ticks_per_image=2,
            percentage_speed_difference=50,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        # Speed-limit
        ExperimentSettings(
            turns=3,
            ticks_per_image=2,
            percentage_speed_difference=0,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        # 50% above speed-limit
        ExperimentSettings(
            turns=3,
            ticks_per_image=2,
            percentage_speed_difference=-50,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
        # 100% above speed-limit
        ExperimentSettings(
            turns=3,
            ticks_per_image=2,
            percentage_speed_difference=-100,
            camera_rigs=[
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=-10)), camera_settings=camera_settings[1]),
                CameraRig(transform=carla.Transform(carla.Location(z=3.0),
                                                    carla.Rotation(yaw=10)), camera_settings=camera_settings[1]),
            ]
        ),
    ],
)
