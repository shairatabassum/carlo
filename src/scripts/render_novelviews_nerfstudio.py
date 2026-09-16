"""
Usage:
    conda activate nerfstudio
    python render_novelviews_nerfstudio.py --config outputs/.../config.yml --mode all
    python render_novelviews_nerfstudio.py --config outputs/.../config.yml --mode translation --shift 2.0
    python render_novelviews_nerfstudio.py --config outputs/.../config.yml --mode translation --shift 4.0
    python render_novelviews_nerfstudio.py --config outputs/.../config.yml --mode rotation --angle 22.5
"""
import argparse
import copy
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from nerfstudio.utils.eval_utils import eval_setup


# ---------------------------------------------------------------------------
# Geometry (mirrors render_novelviews_carla.py, adapted to nerfstudio's own
# camera convention: c2w columns = [right, up, back]; camera left = -col0,
# camera/world up ~= +col1, no OpenCV-style negation needed here).
# ---------------------------------------------------------------------------

def mean_world_up(c2ws: np.ndarray) -> np.ndarray:
    up = c2ws[:, :3, 1].mean(axis=0)
    return up / np.linalg.norm(up)


def axis_angle_rotation(axis: np.ndarray, deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    nx, ny, nz = axis
    return np.array([
        [c + (1-c)*nx*nx,    (1-c)*nx*ny - s*nz, (1-c)*nx*nz + s*ny],
        [(1-c)*ny*nx + s*nz, c + (1-c)*ny*ny,    (1-c)*ny*nz - s*nx],
        [(1-c)*nz*nx - s*ny, (1-c)*nz*ny + s*nx, c + (1-c)*nz*nz],
    ])


def shift_camera(camera, shift_m: float, dataparser_scale: float):
    c = copy.deepcopy(camera)
    c2w = c.camera_to_worlds[0].cpu().numpy()
    lat = -c2w[:3, 0]               # camera left, this frame
    lat /= np.linalg.norm(lat)
    c2w_new = c2w.copy()
    c2w_new[:3, 3] += (shift_m * dataparser_scale) * lat
    c.camera_to_worlds = torch.as_tensor(c2w_new, dtype=camera.camera_to_worlds.dtype)[None]
    return c


def rotate_camera(camera, R: np.ndarray):
    c = copy.deepcopy(camera)
    c2w = c.camera_to_worlds[0].cpu().numpy()
    c2w_new = c2w.copy()
    c2w_new[:3, :3] = R @ c2w[:3, :3]
    c.camera_to_worlds = torch.as_tensor(c2w_new, dtype=camera.camera_to_worlds.dtype)[None]
    return c


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_front_cameras(pipeline, cam_pattern: str):
    """Merge train+eval splits back into one continuous trajectory, filtered to
    the front camera, sorted by frame index."""
    dm = pipeline.datamanager
    entries = []
    for dataset in (dm.train_dataset, dm.eval_dataset):
        if dataset is None:
            continue
        filenames = dataset._dataparser_outputs.image_filenames
        for i, f in enumerate(filenames):
            if cam_pattern in Path(f).as_posix():
                # works for both "cam_0/000000.png" (stem "000000") and flat
                # "000000_0.png" (stem "000000_0") layouts -- frame number is
                # always the leading digit run before any "_camid" suffix.
                frame_idx = int(Path(f).stem.split("_")[0])
                entries.append((frame_idx, dataset.cameras[i:i+1]))
    if not entries:
        raise RuntimeError(
            f"No frames matched cam_pattern='{cam_pattern}' in the dataset's image paths. "
            "Pass --cam-pattern to match your dataset's naming convention."
        )
    entries.sort(key=lambda e: e[0])
    print(f"Found {len(entries)} frames for cam_pattern='{cam_pattern}'")
    return [c for _, c in entries]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_frame(camera, pipeline):
    outputs = pipeline.model.get_outputs_for_camera(camera)
    return outputs["rgb"]  # (H, W, 3), float in [0, 1]


def render_and_save(cameras, pipeline, out_dir: Path, fps: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, camera in enumerate(tqdm(cameras, desc=out_dir.name)):
        frame_path = out_dir / f"{i:06d}.png"
        if frame_path.exists():
            try:
                frames.append(imageio.imread(frame_path))
                continue
            except Exception as e:
                # e.g. truncated file from a process killed mid-write -- re-render it
                print(f"  WARNING: {frame_path} unreadable ({e}), re-rendering it")
        with torch.no_grad():
            rgb = render_frame(camera, pipeline)
        torchvision.utils.save_image(rgb.permute(2, 0, 1), frame_path)
        frames.append((rgb.cpu().numpy() * 255).astype(np.uint8))
    video_path = out_dir.parent / f"{out_dir.name}.mp4"
    try:
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264")
    except Exception as e:
        print(f"  WARNING: video write failed ({e}); frame PNGs are still saved.")
    print(f"  frames -> {out_dir}/")
    print(f"  video  -> {video_path}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_translation(front_cameras, pipeline, out_root: Path, shift_m: float, dataparser_scale: float, fps: int):
    for sign, label in [(+1, f"{shift_m:g}m")]:
        shifted = [shift_camera(c, sign * shift_m, dataparser_scale) for c in front_cameras]
        render_and_save(shifted, pipeline, out_root / label, fps)


def run_rotation(front_cameras, pipeline, out_root: Path, world_up: np.ndarray, angle_deg: float, fps: int):
    for sign, label in [(+1, f"{angle_deg:g}deg")]:
        R = axis_angle_rotation(world_up, sign * angle_deg)
        rotated = [rotate_camera(c, R) for c in front_cameras]
        render_and_save(rotated, pipeline, out_root / label, fps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Path to a trained nerfstudio config.yml")
    ap.add_argument("--mode", choices=["translation", "rotation", "all"], default="all")
    ap.add_argument("--shift", type=float, nargs="+", default=[2.0],
                     help="One or more lateral shifts in meters (used with --mode translation), e.g. --shift 2 4 6")
    ap.add_argument("--angle", type=float, default=22.5, help="Rotation in degrees (used with --mode rotation)")
    ap.add_argument("--cam-pattern", default="cam_0", help="Substring identifying the front camera's file paths")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--output-dir", type=Path, default=None, help="default: <config's output dir>/novel_views")
    args = ap.parse_args()

    def fix_checkpoint_dir(config):
        checkpoint_dir = args.config.resolve().parent / config.relative_model_dir
        config.get_checkpoint_dir = lambda: checkpoint_dir
        return config

    _, pipeline, _, _ = eval_setup(args.config, test_mode="test", update_config_callback=fix_checkpoint_dir)
    front_cameras = get_front_cameras(pipeline, args.cam_pattern)

    dataparser_scale = pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    print(f"dataparser_scale: {dataparser_scale} (real meters -> trained-space units)")

    c2ws = np.stack([c.camera_to_worlds[0].cpu().numpy() for c in front_cameras])
    world_up = mean_world_up(c2ws)
    print(f"World up: {world_up.round(4)} (lateral direction is computed per-frame)")

    out_root = args.output_dir or (args.config.parent / "novel_views")

    if args.mode in ("translation", "all"):
        shifts = args.shift if args.mode == "translation" else [2.0, 4.0]
        for shift_m in shifts:
            print(f"\n--- Translation +-{shift_m} m ---")
            run_translation(front_cameras, pipeline, out_root, shift_m, dataparser_scale, args.fps)

    if args.mode in ("rotation", "all"):
        angle_deg = args.angle if args.mode == "rotation" else 22.5
        print(f"\n--- Rotation +-{angle_deg} deg ---")
        run_rotation(front_cameras, pipeline, out_root, world_up, angle_deg, args.fps)


if __name__ == "__main__":
    main()
