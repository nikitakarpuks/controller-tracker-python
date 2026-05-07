from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    """Load images from a flat directory without labels"""

    def __init__(self, root_dir, crops_per_cam: dict):
        """
        Args:
            root_dir: Path to directory containing images
            crops_per_cam: {cam_idx: (left, top, right, bottom)}
        """
        self.root_dir = Path(root_dir)
        self.crops_per_cam = crops_per_cam

        self.image_paths = sorted(self.root_dir.glob("*.png"))
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        cam_images = {
            cam_idx: np.array(image.crop(crop))
            for cam_idx, crop in self.crops_per_cam.items()
        }
        return img_path, cam_images


def get_data(cfg):
    crops = get_crop_coordinates(cfg)
    dataset = ImageDataset(root_dir=cfg["root"], crops_per_cam=crops)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    return dataloader


def get_crop_coordinates(cfg) -> dict:
    """Returns {cam_idx: (left, top, right, bottom)} for each selected camera."""
    part_width = cfg["img_width"] // cfg["total_cameras_number"]
    top = 1  # top pixel row has exposure/gain values encoded, skip it
    bottom = cfg["img_height"]
    return {
        cam_idx: (cam_idx * part_width, top, (cam_idx + 1) * part_width, bottom)
        for cam_idx in cfg["selected_cameras"]
    }