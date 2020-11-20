import numpy as np

from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from pathlib import Path
from dataset import Dataset
from typing import List

import config
from utils import fix_wmap, class_weights

c = config.Config()


class AugmentedSequence(Sequence):
    def __init__(self, dataset: Dataset, aug: iaa.Sequential, scale=1.0):

        self.data = []
        for pid in dataset.ids:
            im = dataset.load_image(pid, scale)
            self.data.append(
                (
                    im,
                    SegmentationMapsOnImage(dataset.get_mask(pid, scale), im.shape),
                    HeatmapsOnImage(
                        dataset.get_weight_map(pid, scale).astype(np.float32),
                        im.shape,
                        max_value=10,
                    ),
                )
            )

        self.batch_size = c.BATCH_SIZE
        self.aug = aug
        self.data_size = len(dataset.ids)
        self.img_width = c.WIDTH
        self.img_height = c.HEIGHT
        self.class_weights = class_weights(dataset, self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        im, mask, wmap = self.data[idx % self.data_size]

        augims, augmasks, augwmaps = zip(
            *[
                self.aug(image=im, segmentation_maps=mask, heatmaps=wmap)
                for i in range(self.batch_size)
            ]
        )

        augmasks = [to_categorical(m.get_arr(), num_classes=3) for m in augmasks]
        augwmaps = [
            fix_wmap(w.get_arr(), m, self.class_weights)
            for w, m in zip(augwmaps, augmasks)
        ]

        augims = np.asarray(augims)
        augmasks = np.asarray(augmasks, dtype=np.float32)
        augwmaps = np.asarray(augwmaps, dtype=np.float32)

        augmasks = np.asarray(augmasks).reshape(
            (self.batch_size, self.img_height, self.img_width, 3)
        )

        return (augims, augwmaps), augmasks

    def __repr__(self):
        tmp_str = "\n\t".join(self.patient_ids)
        return (
            "AugmentedSequence:\n"
            + f"batch_size = {self.batch_size}\n"
            + f"ids:\t{tmp_str}"
        )
