import logging
import os
from enum import Enum

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset as TDataset
from torchvision import transforms

from net.utils import get_files

logger = logging.getLogger(__name__)


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1


class Transformer:
    def __init__(self):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def apply_base_transform(self, path):
        return self.base_transform(Image.open(path))

    def __call__(self, image1, image2=None):
        image1 = self.apply_base_transform(image1)
        i, j, h, w = transforms.RandomCrop.get_params(image1, output_size=(256, 256))
        image1 = TF.crop(image1, i, j, h, w)
        if image2 is not None:
            image2 = self.apply_base_transform(image2)
            image2 = TF.crop(image2, i, j, h, w)
            return image1, image2

        return image1


class Dataset(TDataset):
    def __init__(
        self,
        root_dir="",
        dataset="maps",
        mode="train",
        direction=Direction.FORWARD,
        max_items=None,
    ):
        self.direction = direction
        self.transformer = Transformer()
        file_path = os.path.join(root_dir, dataset)
        file_path_mode = os.path.join(file_path, mode)

        self.lfiles = get_files(file_path_mode, "l_")
        self.rfiles = get_files(file_path_mode, "r_")

        if max_items is not None:
            self.lfiles = self.lfiles[:max_items]
            self.rfiles = self.rfiles[:max_items]

        assert len(self.lfiles) == len(self.rfiles)
        logger.info(f"Dataset {dataset}/{mode} was created. Size: {len(self.lfiles)}")

    def __len__(self):
        return len(self.lfiles)

    def __getitem__(self, item):
        input, target = self.transformer(self.lfiles[item], self.rfiles[item])
        if self.direction == Direction.BACKWARD:
            input, target = target, input
        elif self.direction != Direction.FORWARD:
            raise RuntimeError(f"Unknown direction {self.direction}")

        return {"input": input, "target": target}
