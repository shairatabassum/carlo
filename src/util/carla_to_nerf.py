import carla
import numpy as np

def carla_to_nerf(camera_transform: carla.Transform):
    """
        Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """
    unreal_location = camera_transform.location
    unreal_rotation = camera_transform.rotation

    # Convert from Unreal Engine to Blender coordinate system
    # From Carla Docs: Warning: The declaration order is different in CARLA (pitch,yaw,roll), and in the Unreal Engine Editor (roll,pitch,yaw). When working in a build from source, don't mix up the axes' rotations.
    blender_matrix = carla.Transform(
        carla.Location(x=-unreal_location.z, y=unreal_location.x, z=unreal_location.y),
        carla.Rotation(pitch=unreal_rotation.yaw + 90, yaw=unreal_rotation.roll + 90, roll=unreal_rotation.pitch)
    )
    
    return blender_matrix.get_matrix()

def carla_to_nerf_back(camera_transform: carla.Transform):
    """
        Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """
    unreal_location = camera_transform.location
    unreal_rotation = camera_transform.rotation

    # Convert from Unreal Engine to Blender coordinate system
    # From Carla Docs: Warning: The declaration order is different in CARLA (pitch,yaw,roll), and in the Unreal Engine Editor (roll,pitch,yaw). When working in a build from source, don't mix up the axes' rotations.
    blender_matrix = carla.Transform(
        carla.Location(x=-unreal_location.z, y=unreal_location.x, z=unreal_location.y),
        carla.Rotation(pitch=unreal_rotation.yaw + 90, yaw=unreal_rotation.roll + 90, roll=unreal_rotation.pitch)
    )
    
    return blender_matrix

def carla_to_nerf_2(camera_transform: carla.Transform):
    """
        Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """
    unreal_location = camera_transform.location

    matrix = np.array(camera_transform.get_matrix())
    matrix[:3, 3] = np.array([-unreal_location.z, unreal_location.x, unreal_location.y])

    # Convert from Unreal Engine to Blender coordinate system
    # 1. Go from left-handed to right-handed coordinate system
    r_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [ 0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    matrix = r_matrix @ matrix

    # 2. Rotate 180 degrees around the y-axis
    r_matrix = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [ 0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    matrix = r_matrix @ matrix

    # 3. Rotate 90 degrees around the z-axis
    r_matrix = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    matrix = r_matrix @ matrix


    # 2. Rotate 90 degrees around the x-axis
    r_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [ 0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    matrix = r_matrix @ matrix

    # Should be the same as carla_to_nerf
    return matrix.tolist()

def carla_to_nerf_3(camera_transform: carla.Transform):
    """
        Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """
    unreal_location = camera_transform.location
    unreal_rotation = camera_transform.rotation

    # Convert from Unreal Engine to Blender coordinate system
    # From Carla Docs: Warning: The declaration order is different in CARLA (pitch,yaw,roll), and in the Unreal Engine Editor (roll,pitch,yaw). When working in a build from source, don't mix up the axes' rotations.
    blender_matrix = carla.Transform(
        carla.Location(x=-unreal_location.y, y=unreal_location.x, z=unreal_location.z),
        carla.Rotation(roll=unreal_rotation.roll, pitch=unreal_rotation.pitch, yaw=unreal_rotation.yaw + 90)
    )
    
    return blender_matrix.get_matrix()