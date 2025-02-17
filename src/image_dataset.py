from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import Dataset, DataLoader


def _list_image_files_recursively(data_dir):
    results = []

    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):

    def __init__(self, resolution, image_paths):
        super().__init__()
        self.resolution = resolution
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.image[idx]

        with bf.open(path, "r") as f:
            pil_image = Image.open(f)
            pil_image.load()

        while min(*pil_image.size) < 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        return np.transpose(arr, [2, 0, 1]), {}


def load_data(*, data_dir, image_size, batch_size, deterministic=False):

    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)

    dataset = ImageDataset(image_size, all_files)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

    while True:
        yield from loader
