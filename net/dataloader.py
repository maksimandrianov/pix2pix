import logging
import os
from enum import Enum

from PIL import Image
from torch.utils.data import Dataset as TDataset
from torchvision import transforms

from net.utils import get_files

logger = logging.getLogger(__name__)


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1


class Transformer:
    def __init__(self, test_mode):
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop((256, 256) if test_mode else (512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, path):
        return self.transform(Image.open(path))


class Dataset(TDataset):
    def __init__(
        self,
        root_dir="",
        dataset="maps",
        mode="train",
        direction=Direction.FORWARD,
        max_items=None,
        test_mode=False,
    ):
        self.direction = direction
        self.transformer = Transformer(test_mode)
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
        input = self.transformer(self.lfiles[item])
        target = self.transformer(self.rfiles[item])
        if self.direction == Direction.BACKWARD:
            input, target = target, input
        elif self.direction != Direction.FORWARD:
            raise RuntimeError(f"Unknown direction {self.direction}")

        return {"input": input, "target": target}
