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
import pandas as pd

memory = Memory("./cache", verbose=0)


class Dataset:
    def __init__(self, path: Path):
        self.path = path
        self.image_dir = path / "images"
        self.ids = np.array([Path(f).stem for f in os.listdir(self.image_dir)])
        self.image_shape = self.load_image(self.ids[0]).shape

    @property
    def size(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.load_image(self.ids[idx])

    def file_name(self, image_id):
        return str(self.image_dir / f"{image_id}.png")

    def load_image(self, image_id):
        return imread(self.image_dir / f"{image_id}.png")

    def make_split(self, factor=0.8):
        np.random.seed(0)
        train_index = np.random.rand(self.size) < 0.8

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

    def get_mask(self, patient_id: str):
        return np.array(utils.get_boundary_mask(utils.get_mask(patient_id)), dtype=np.int8)

    def get_weight_map(self, patient_id: str):
        return utils.get_weight_map(patient_id)


class Swebcg(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/swebcg/")

        with open(self.path / "annotations.json") as f:
            self.annotations = json.load(f)

    def get_annotation(self, patient_id: str):
        return [cell["vertices"] for cell in self.annotations[patient_id]]

    def get_dataframe(self):
        data = []
        for key, item in utils.parse_annotations(self.annotations).items():
            if key in self.ids:
                for cell in item:
                    data.append({'vertices' : cell['vertices'],
                                'class' : cell['class'],
                                'image_id' : key})
        return pd.DataFrame(data)


class Bns(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / "data/bns/")

    def load_image(self, image_id):
        return imread(self.image_dir / f"{image_id}.png")[..., 0:3]  # skipping alpha

    def get_mask(self, image_id):
        mask = nib.load(self.path / f"masks/{image_id}.nii.gz").get_fdata() > 0
        mask = np.transpose(mask, axes=(1, 0, 2))[..., 0]
        return mask.astype(np.int8)

    def get_weight_map(self, image_id):
        return utils.get_weight_map_bns(image_id)

    def get_annotation(self, image_id):
        anno = nib.load(self.path / f"masks/{image_id}.nii.gz").get_fdata()
        poly = Mask(anno).polygons()
        return poly.points


class Quip(Dataset):
    def __init__(self):
        self.path = Path(__file__).parent.parent / "data/quip"
        self.image_dir = self.path / "images"
        self.anno_dir = self.path / "annotations"
        self.slides = {}
        for root, _, files in os.walk(self.image_dir):
            for name in files:
                if name[-3:] == "svs":
                    self.slides[_submitter_id(name)] = os.path.join(root, name)

        self.annotations = {}
        for root, dirs, files in os.walk(self.anno_dir):
            for name in files:
                if name[-3:] == "csv":
                    slide_id = _submitter_id(Path(root).stem.upper())
                    if slide_id in self.annotations:
                        self.annotations[slide_id].append(
                            {"file_name": name, "region": self._region(name)}
                        )
                    else:
                        self.annotations[slide_id] = [
                            {"file_name": name, "region": self._region(name)}
                        ]
        self.ids = []
        for key, item in self.annotations.items():
            if key in self.slides:
                self.ids.append[key]

    @property
    def _anno_ids(self):
        return [Path(f.upper()).stem[:-8] for f in os.listdir(self.anno_dir)]

    @property
    def _slide_ids(self):
        return [f for f in os.walk(self.image_dir)]

    def _region(self, file_name):
        x, y, width, height = re.findall(r"(\d*)_(\d*)_(\d*)_(\d*)", file_name)[0]
        return int(x), int(y), int(width), int(height)


def _submitter_id(file_name : str):
    return  re.findall("[^.]*", file_name)[0]
        



if __name__ == "__main__":
    quip = Quip()
    print(list(quip.slides.keys())[0])

    print(list(quip.annotations.keys())[0])
    print(quip.ids)
