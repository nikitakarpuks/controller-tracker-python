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
    """
    crops = get_crop_coordinates(cfg)
    root_dir = Path(cfg["root"])
    image_paths = sorted(root_dir.glob("*.png"))
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


def count_images(cfg) -> int:
    """Number of frames get_data(cfg) will yield — lets a caller pass tqdm(..., total=...)
    without having to start iterating first (get_data is a generator, so it has no __len__)."""
    return len(list(Path(cfg["root"]).glob("*.png")))


def get_crop_coordinates(cfg) -> dict:
    """Returns {cam_idx: (left, top, right, bottom)} for each selected camera."""
    part_width = cfg["img_width"] // cfg["total_cameras_number"]
    top = 1 if cfg.get("has_technical_row", True) else 0
    bottom = cfg["img_height"]
    return {
        cam_idx: (cam_idx * part_width, top, (cam_idx + 1) * part_width, bottom)
        for cam_idx in cfg["selected_cameras"]
    }
