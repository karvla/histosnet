import numpy as np
from keras.utils.np_utils import to_categorical
from skimage.transform import rescale
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import config
import utils
from dataset import Dataset
from time import time
import tensorflow as tf
import random

c = config.Config


class AugmentedDataset(tf.data.Dataset):
    def __new__(self, dataset, aug, num_samples=3, scale=1.0):
        return tf.data.Dataset.from_generator(
            lambda: self._generator(self, dataset, aug, num_samples, scale),
            output_types=((tf.dtypes.uint8, tf.dtypes.float32), tf.dtypes.uint8),
        )

    def _generator(self, dataset, aug, num_samples, scale):
        def _fix_wmap_dim(wmap):
            wmap = wmap.ravel()
            wmap = np.array((np.ones(wmap.shape), wmap, wmap)).T
            wmap = wmap.reshape((c.HEIGHT, c.HEIGHT, 3))
            return wmap

        for imid in random.choices(dataset.ids, k=num_samples):

            image = dataset.load_image(imid, scale)
            mask = dataset.get_mask(imid, scale)

            s = 4
            split_h = lambda img: np.array_split(img, s, axis=1)
            split_v = lambda img: np.array_split(img, s, axis=0)
            for imrow, maskrow in zip(split_h(image), split_h(mask)):
                for im, mas in zip(split_v(imrow), split_v(maskrow)):
                    mas = SegmentationMapsOnImage(mas, im.shape)

                    aug_pairs = [
                        aug(image=im, segmentation_maps=mas) for i in range(100)
                    ]
                    aug_pairs = sorted(aug_pairs, key=lambda x: np.sum(x[1].get_arr()))[
                        -c.BATCH_SIZE :
                    ]

                    augims, augmasks = zip(*aug_pairs)

                    augwmaps = [
                        _fix_wmap_dim(utils.unet_weight_map(m.get_arr() > 0, c.WIDTH))
                        for m in augmasks
                    ]
                    augmasks = [
                        to_categorical(m.get_arr(), num_classes=3) for m in augmasks
                    ]

                    augims = np.asarray(augims)
                    augmasks = np.asarray(augmasks, dtype=np.uint8)
                    augwmaps = np.asarray(augwmaps, dtype=np.float32)

                    augmasks = np.asarray(augmasks).reshape(c.BATCH_SHAPE)

                    yield (augims, augwmaps), augmasks
