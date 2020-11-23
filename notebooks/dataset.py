from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import utils
from joblib import Memory
from typing import Optional, List
from copy import copy
from skimage.io import imread
import json
import nibabel as nib
import numpy as np
import regex as re
from imantics import Polygons, Mask
from PIL import Image, ImageDraw, ImageStat
import pandas as pd
from tqdm import tqdm
from openslide import OpenSlide
from itertools import tee
from skimage.transform import rescale
from skimage.morphology import remove_small_objects
from indexing import get_quip_image_index, get_quip_annotation_index
import random
from utils import fix_wmap
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from config import Config
from keras.utils.np_utils import to_categorical
from keras.utils import Sequence

c = Config()
memory = Memory("./cache", verbose=0)


class Dataset(Sequence):
    def __init__(self, path: Path):
        self.path = path
        self.image_dir = path / "images"
        self.ids = np.array([Path(f).stem for f in os.listdir(self.image_dir)])
        self.ending = Path(os.listdir(self.image_dir)[0]).suffix
        self.name = path.stem
        self.size = len(self.ids)
        self.aug_size = self.size
        self.aug = iaa.Sequential(
            [
                iaa.CropToFixedSize(width=c.WIDTH, height=c.HEIGHT),
            ]
        )

    def __len__(self):
        return self.aug_size

    def __getitem__(self, idx):
        image_id = self.ids[idx % len(self.ids) - 1]

        im = self.load_image(image_id)
        mask = self.get_mask(image_id)
        wmap = self.get_weight_map(image_id)

        mask = SegmentationMapsOnImage(mask, im.shape)
        wmap = HeatmapsOnImage(
            wmap.astype(np.float32),
            im.shape,
        )

        augims, augmasks, augwmaps = zip(
            *[
                self.aug(image=im, segmentation_maps=mask, heatmaps=wmap)
                for i in range(c.BATCH_SIZE)
            ]
        )

        augmasks = [to_categorical(m.get_arr(), num_classes=3) for m in augmasks]
        augwmaps = [fix_wmap(w.get_arr()) for w, m in zip(augwmaps, augmasks)]

        augims = np.asarray(augims)
        augmasks = np.asarray(augmasks, dtype=np.float32)
        augwmaps = np.asarray(augwmaps, dtype=np.float32)

        # augmasks = np.asarray(augmasks).reshape(
        #    (c.BATCH_SHAPE)
        # )

        # return (augims, np.ones_like(augims)), augmasks
        return (augims, augwmaps), augmasks

    def file_name(self, image_id):
        return str(self.image_dir / f"{image_id}{self.ending}")

    def load_image(self, image_id, scale=1.0):
        return self._rescale(imread(self.image_dir / f"{image_id}{self.ending}"), scale)

    def _rescale(self, img, scale):
        multichannel = len(img.shape) > 2
        og_type = img.dtype
        if scale > 1.0:
            img = rescale(img, scale, multichannel=multichannel, preserve_range=True)
        elif scale < 1.0:
            if not multichannel and img.dtype != np.float32:
                img = remove_small_objects(img.astype(np.bool), 10)
            img = rescale(
                img,
                scale,
                multichannel=multichannel,
                anti_aliasing=True,
                preserve_range=True,
            )
        return img.astype(og_type)

    def make_split(self, factor=0.8):
        np.random.seed(0)
        train_index = np.random.rand(self.size) < factor

        train_set = copy(self)
        train_set.ids = self.ids[train_index]

        test_set = copy(self)
        test_set.ids = self.ids[~train_index]
        return train_set, test_set


class Monuseg(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/monuseg/")

    def get_annotation(self, patient_id: str):
        return utils.get_annotation_monuseg(patient_id)

    def get_mask(self, patient_id: str, scale=1.0):
        mask = self._rescale(utils.get_mask(patient_id), scale)
        return utils.get_boundary_mask(mask).astype(np.int8)

    def get_weight_map(self, patient_id: str, scale=1.0):
        return self._rescale(utils.get_weight_map(patient_id), scale)


class TNBC1(Dataset):
    """ 80 images with some annotated cells """

    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/tnbc1/")

        with open(self.path / "annotations.json") as f:
            self.annotations = json.load(f)

    def _bbox_to_slice(self, bbox):
        return (
            slice(int(bbox[0][1]), int(bbox[1][1]), None),
            slice(int(bbox[0][0]), int(bbox[1][0]), None),
        )

    def _generate_mask(self, vertices):
        img = Image.new("L", (3000, 3000), 0)
        ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
        (left, upper), (right, lower) = utils.bounding_box(vertices)
        img = img.crop((left, upper, right + 1, lower + 1))
        return np.asarray(img)

    def get_annotation(self, patient_id: str):
        return [cell["vertices"] for cell in self.annotations[patient_id]]

    def get_dataframe(self):
        data = []
        for key, item in utils.parse_annotations(self.annotations).items():
            for cell in item:
                data.append(
                    {
                        "class": cell["class"],
                        "image_id": key,
                        "obj": self._bbox_to_slice(
                            utils.bounding_box(cell["vertices"])
                        ),
                        "mask": self._generate_mask(cell["vertices"]),
                    }
                )
        return pd.DataFrame(data)


class TNBC2(Dataset):
    """ 530 images without annotations """

    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/tnbc2/", training=False)


class TNBCWSI(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/tnbc_wsi/")


class Bns(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/bns/")

    def load_image(self, image_id, scale=1.0):
        img = imread(self.image_dir / f"{image_id}.png")[..., 0:3]  # skipping alpha
        return self._rescale(img, scale)

    def get_mask(self, image_id, scale=1.0):
        mask = nib.load(self.path / f"masks/{image_id}.nii.gz").get_fdata() > 0
        mask = np.transpose(mask, axes=(1, 0, 2))[..., 0]
        return utils.get_boundary_mask(self._rescale(mask.astype(np.uint8), scale))

    def get_weight_map(self, image_id, scale=1.0):
        return self._rescale(utils.get_weight_map_bns(image_id), scale)

    def get_annotation(self, image_id):
        anno = nib.load(self.path / f"masks/{image_id}.nii.gz").get_fdata()
        poly = Mask(anno).polygons()
        return poly.points


class Quip(Dataset):
    def __init__(self, n_cells_threshold=2000):
        self.path = Path(__file__).parent.parent / "data/quip"
        self.image_dir = self.path / "images"
        self.anno_dir = self.path / "annotations"
        self.slides = get_quip_image_index(self.image_dir)
        self.annotations = get_quip_annotation_index(self.anno_dir)

        self.ids = []

        for key, item in self.annotations.items():
            if key in self.slides:
                self.ids.extend(
                    [
                        (key, region)
                        for region, value in item.items()
                        if value["n_cells"] > n_cells_threshold
                    ]
                )

    @property
    def _anno_ids(self):
        return [Path(f.upper()).stem[:-8] for f in os.listdir(self.anno_dir)]

    @property
    def _slide_ids(self):
        return [f for f in os.walk(self.image_dir)]

    def load_image(self, image_id, scale=1.0):
        slide_id, region = image_id
        img = np.array(
            OpenSlide(self.slides[slide_id]).read_region(region[:2], 0, region[2:])
        )[..., :3]

        return self._rescale(img, scale)

    def get_annotation(self, image_id):
        slide_id, region = image_id
        path = self.anno_dir / self.annotations[slide_id][region]["path"]
        anno = list(pd.read_csv(path)["Polygon"])
        x0, y0, width, height = region
        annotations = []
        for polygon in anno:
            polygon = polygon[1:-1].split(":")
            annotations.append(
                [
                    (float(x) - x0, float(y) - y0)
                    for x, y in zip(polygon[0::2], polygon[1::2])
                ]
            )
        return annotations

    def get_mask(self, image_id, scale=1.0):
        _, (_, _, width, height) = image_id
        mask = utils.generate_mask(self.get_annotation(image_id), (width, height))
        mask = self._rescale(mask, scale)
        return utils.get_boundary_mask(mask).astype(np.int8)

    def get_weight_map(self, image_id):
        _, (_, _, width, height) = image_id
        return np.ones((width, height), dtype=np.float32)


if __name__ == "__main__":
    quip = Quip()
    print(list(quip.slides.keys())[0])

    print(list(quip.annotations.keys())[0])
    print(quip.ids)
