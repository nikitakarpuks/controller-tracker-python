from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    """Load images from a flat directory without labels"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to directory containing images
            transform: Optional transform to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get all image files
        self.image_paths = []
        extensions = ["*.png"]

        for ext in extensions:
            self.image_paths.extend(self.root_dir.glob(ext))
            self.image_paths.extend(self.root_dir.glob(ext.upper()))

        # Sort for consistent ordering
        self.image_paths = sorted(self.image_paths)

        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image  # Return only image, no label

class StaticCrop:
    """Static crop to specific coordinates"""

    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __call__(self, img):
        return img.crop((self.left, self.top, self.right, self.bottom))


class ToNumpy:
    """Convert PIL Image or Tensor to numpy array"""

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return pic
        # Convert PIL Image to numpy
        return np.array(pic)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def get_data(cfg):
    data_transform = get_transform(cfg)

    dataset = ImageDataset(
        root_dir=cfg["root"],
        transform=data_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch)

    return dataloader

def get_transform(cfg):
    crop_bbox = get_crop_coordinates(cfg)
    data_transform = transforms.Compose([
        StaticCrop(*crop_bbox),
        # transforms.ToTensor()
        ToNumpy()
    ])
    return data_transform

def get_crop_coordinates(cfg):
    """Get crop coordinates based on cameras we want to use.
    The image is horizontally split into equal parts, each part corresponds to a camera."""

    part_width = cfg["img_width"] // cfg["total_cameras_number"]
    selected_cameras = cfg["selected_cameras"]
    left = selected_cameras[0] * part_width
    right = (selected_cameras[-1] + 1) * part_width
    top = 1  # top pixel row  has some exposure and gain values encoded, ignoring it
    bottom = cfg["img_height"]

    return left, top, right, bottom
