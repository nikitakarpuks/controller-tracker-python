from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2


def _load_frame(img_path: Path, crops_per_cam: dict):
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    cam_images = {
        cam_idx: image[top:bottom, left:right].copy()
        for cam_idx, (left, top, right, bottom) in crops_per_cam.items()
    }
    return img_path, cam_images


def _load_frame_per_camera(img_paths: dict, top: int):
    """img_paths: {cam_idx: Path}, one already-per-camera file per frame."""
    cam_images = {
        cam_idx: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[top:].copy()
        for cam_idx, path in img_paths.items()
    }
    representative_path = next(iter(img_paths.values()))
    return representative_path, cam_images


def _apply_frame_range(paths: list, cfg) -> list:
    """Slices an alphabetically-sorted image list down to cfg["frame_range"], if set.

    `lower` (inclusive, 0-based) and `upper` (exclusive) follow normal Python slice
    semantics; either may be omitted/null to leave that end unbounded.
    """
    frame_range = cfg.get("frame_range") or {}
    return paths[frame_range.get("lower"):frame_range.get("upper")]


def _camera_dir(cfg, cam_idx) -> Path:
    root_dir = Path(cfg["root"])
    folder = cfg.get("camera_folder_pattern", "cam{idx}").format(idx=cam_idx)
    images_subdir = cfg.get("images_subdir", "")
    return root_dir / folder / images_subdir if images_subdir else root_dir / folder


def _per_camera_image_lists(cfg) -> dict:
    """Returns {cam_idx: [sorted image paths]} for the "per_camera_folders" layout.

    Frames are paired across folders by sorted position, not by filename — folders
    are only required to hold the same number of images, not matching filenames.
    """
    lists = {cam_idx: _apply_frame_range(sorted(_camera_dir(cfg, cam_idx).glob("*.png")), cfg)
             for cam_idx in cfg["selected_cameras"]}
    counts = {cam_idx: len(paths) for cam_idx, paths in lists.items()}
    if len(set(counts.values())) > 1:
        raise ValueError(
            f"data.layout='per_camera_folders' requires the same image count in every "
            f"camera folder, got {counts}"
        )
    return lists


def get_data(cfg):
    """
    Yields one [(img_path, cam_images)] batch per frame, sorted by filename —
    same shape callers already unpack via `batch[0][0], batch[0][1]`.

    Decodes one frame ahead on a background thread while the caller processes
    the current frame: PNG decode is pure CPU-bound decompression (~7ms/frame,
    confirmed to release the GIL), smaller than the rest of the per-frame
    pipeline (blob detection + pose search + logging, ~20ms), so a single
    frame of lookahead is enough to fully hide it — reading further ahead
    wouldn't help, it would just buffer frames faster than the caller
    consumes them.

    Supports two `cfg["layout"]` values:
    - "concatenated" (default): one folder, every camera's view packed side-by-side
      into a single wide image per frame (see get_crop_coordinates).
    - "per_camera_folders": one subfolder per camera (see _camera_dir), each holding
      the same number of images; frames are paired by sorted position.

    In both layouts, `cfg["frame_range"]` (see _apply_frame_range) can restrict
    processing to a slice of the alphabetically-sorted image list(s).
    """
    if cfg.get("layout", "concatenated") == "per_camera_folders":
        yield from _get_data_per_camera(cfg)
        return

    crops = get_crop_coordinates(cfg)
    root_dir = Path(cfg["root"])
    image_paths = _apply_frame_range(sorted(root_dir.glob("*.png")), cfg)
    print(f"Found {len(image_paths)} images in {root_dir}")

    if not image_paths:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_frame, image_paths[0], crops)
        for idx in range(len(image_paths)):
            img_path, cam_images = future.result()
            if idx + 1 < len(image_paths):
                future = executor.submit(_load_frame, image_paths[idx + 1], crops)
            yield [(img_path, cam_images)]


def _get_data_per_camera(cfg):
    image_lists = _per_camera_image_lists(cfg)
    n_frames = len(next(iter(image_lists.values()))) if image_lists else 0
    print(f"Found {n_frames} frames across {len(image_lists)} camera folders under {cfg['root']}")

    if n_frames == 0:
        return

    top = 1 if cfg.get("has_technical_row", True) else 0

    def frame_paths(idx):
        return {cam_idx: paths[idx] for cam_idx, paths in image_lists.items()}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_frame_per_camera, frame_paths(0), top)
        for idx in range(n_frames):
            img_path, cam_images = future.result()
            if idx + 1 < n_frames:
                future = executor.submit(_load_frame_per_camera, frame_paths(idx + 1), top)
            yield [(img_path, cam_images)]


def count_images(cfg) -> int:
    """Number of frames get_data(cfg) will yield — lets a caller pass tqdm(..., total=...)
    without having to start iterating first (get_data is a generator, so it has no __len__)."""
    if cfg.get("layout", "concatenated") == "per_camera_folders":
        image_lists = _per_camera_image_lists(cfg)
        return len(next(iter(image_lists.values()))) if image_lists else 0
    return len(_apply_frame_range(sorted(Path(cfg["root"]).glob("*.png")), cfg))


def get_crop_coordinates(cfg) -> dict:
    """Returns {cam_idx: (left, top, right, bottom)} for each selected camera.

    Only used for the "concatenated" layout, where every camera's image is packed
    side-by-side into a single wide file per frame.
    """
    part_width = cfg["img_width"] // cfg["total_cameras_number"]
    top = 1 if cfg.get("has_technical_row", True) else 0
    bottom = cfg["img_height"]
    return {
        cam_idx: (cam_idx * part_width, top, (cam_idx + 1) * part_width, bottom)
        for cam_idx in cfg["selected_cameras"]
    }
