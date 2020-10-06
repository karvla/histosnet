import numpy as np

from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from pathlib import Path
from skimage.io import imread
from dataset import Dataset
from utils import get_mask, get_weight_map
from typing import List
import config

c = config.Config()
class AugmentedSequence(Sequence):
    def __init__(self, datasets : List[Dataset], aug: iaa.Sequential):

        self.data = []
        for dataset in datasets:
            for pid in dataset.ids:
                im = dataset.load_image(pid)
                self.data.append(
                    (
                        im,
                        SegmentationMapsOnImage(dataset.get_mask(pid), im.shape),
                        HeatmapsOnImage(
                            dataset.get_weight_map(pid).astype(np.float32), im.shape, max_value=10
                        ),
                    )
                )

        self.batch_size = c.BATCH_SIZE
        self.aug = aug
        self.data_size = len(dataset.ids)
        self.img_width = c.WIDTH
        self.img_height = c.HEIGHT

    def __len__(self):
        return self.data_size

    def _fix_wmap_dim(self, wmap):
        wmap = wmap.ravel()
        wmap = np.array((np.ones(wmap.shape), wmap, wmap)).T
        wmap = wmap.reshape((self.img_height, self.img_width, 3))
        return wmap

    def __getitem__(self, idx):
        im, mask, wmap = self.data[idx % self.data_size]

        augims, augmasks, augwmaps = zip(
            *[
                self.aug(image=im, segmentation_maps=mask, heatmaps=wmap)
                for i in range(self.batch_size)
            ]
        )

        augmasks = [to_categorical(m.get_arr(), num_classes = 3) for m in augmasks]
        augwmaps = [self._fix_wmap_dim(m.get_arr()) for m in augwmaps]

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
