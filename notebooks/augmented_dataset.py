import numpy as np
from keras.utils.np_utils import to_categorical
from skimage.transform import rescale
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from skimage.morphology import remove_small_objects
import config
import utils
from dataset import Dataset
from time import time
import tensorflow as tf
import random
from utils import fix_wmap

c = config.Config


class AugmentedDataset(tf.data.Dataset):
    def __new__(self, dataset, aug, class_weights, num_samples=3, scale=1.0):
        return tf.data.Dataset.from_generator(
            lambda: self.generator(
                self, dataset, aug, class_weights, num_samples, scale
            ),
            output_types=((tf.dtypes.uint8, tf.dtypes.float32), tf.dtypes.uint8),
        )

    def generator(self, dataset, aug, class_weights, num_samples, scale):

        imid = random.choice(dataset.ids)
        image = dataset.load_image(imid, scale)

        mask = dataset.get_mask(imid, scale)
        mask = remove_small_objects(mask, 5)

        cw = class_weights

        s = 4
        split_h = lambda img: np.array_split(img, s, axis=1)
        split_v = lambda img: np.array_split(img, s, axis=0)
        for imrow, maskrow in zip(split_h(image), split_h(mask)):
            for im, mas in zip(split_v(imrow), split_v(maskrow)):
                mas = SegmentationMapsOnImage(mas, im.shape)

                for i in range(num_samples):
                    augims, augmasks = zip(
                        *[
                            aug(image=im, segmentation_maps=mas)
                            for i in range(c.BATCH_SIZE)
                        ]
                    )

                    augwmaps = [
                        fix_wmap(
                            utils.unet_weight_map(m.get_arr() > 0, c.WIDTH),
                            to_categorical(m.get_arr(), num_classes=3),
                            cw,
                        )
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
