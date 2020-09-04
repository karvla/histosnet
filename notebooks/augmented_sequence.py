import numpy as np

from keras.utils import Sequence
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from pathlib import Path
from skimage.io import imread

from monuseg_utils import load_image, get_mask, get_weight_map


class AugmentedSequence(Sequence):
    def __init__(
        self,
        patient_ids,
        batch_size,
        aug: iaa.Sequential,
        img_width,
        img_height,
    ):

        self.data = []
        for pid in patient_ids:
            im = load_image(pid)
            self.data.append(
                (
                    im,
                    SegmentationMapsOnImage(im, im.shape),
                    HeatmapsOnImage(get_weight_map(pid).astype(np.float32), im.shape),
                )
            )

        print(np.shape(self.data[0][0]))
        print(np.shape(self.data[0][1]))
        print(np.shape(self.data[0][2]))
        self.batch_size = batch_size
        self.aug = aug
        self.data_size = len(patient_ids)
        self.img_width = img_width
        self.img_height = img_height

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
        augmasks = [m.get_arr() for m in augmasks]
        augwmaps = [m.get_arr() for m in augwmaps]
        print(np.shape(augims))
        print(np.shape(augmasks))
        print(np.shape(augwmaps))

        augims = np.asarray(augims)
        #augmasks = np.asarray(augmasks).reshape(
        #    (self.batch_size, self.img_height, self.img_width, 1)
        #)
        #augwmaps = np.asarray(augwmaps).reshape(
        #    (self.batch_size, self.img_height, self.img_width, 1)
        #)

        return augims, augmasks, augwmaps

    def __repr__(self):
        tmp_str = "\n\t".join(self.patient_ids)
        return (
            "AugmentedSequence:\n"
            + f"batch_size = {self.batch_size}\n"
            + f"ids:\t{tmp_str}"
        )
