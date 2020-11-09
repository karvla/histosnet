import os
import pandas as pd
import regex as re
from pathlib import Path
from joblib import Memory
from tqdm import tqdm

memory = Memory("./cache", verbose=0)


@memory.cache
def get_quip_image_index(image_dir):
    print(f"Indexing images in {image_dir}")
    slides = {}
    cq = pd.read_csv(image_dir.parent / "wsi_quality_control_result.txt")
    good_slides = cq[cq["SegmentationUnacceptableOrNot"] == "0"]["WSI-ID"].unique()

    for root, _, files in tqdm(os.walk(image_dir)):
        for name in files:
            if name[-3:] == "svs" and name in good_slides:
                slides[_submitter_id(name)] = os.path.join(root, name)
    return slides


@memory.cache
def get_quip_annotation_index(anno_dir):
    print(f"Indexing annotations in {anno_dir}")
    annotations = {}
    for root, dirs, files in os.walk(anno_dir):
        for name in files:
            if name[-3:] == "csv":
                slide_id = _submitter_id(Path(root).stem.upper())
                region = _region(name)
                path = os.path.join(root, name)

                if slide_id in annotations:
                    annotations[slide_id][region] = {
                        "path": path,
                        "n_cells": _n_cells(path),
                    }
                else:
                    annotations[slide_id] = {
                        region: {"path": path, "n_cells": _n_cells(path)}
                    }
    return annotations


def _submitter_id(file_name: str):
    return re.findall("[^.]*", file_name)[0]


def _region(file_name):
    x, y, width, height = re.findall(r"(\d*)_(\d*)_(\d*)_(\d*)", file_name)[0]
    return int(x), int(y), int(width), int(height)


def _n_cells(annotation_path):
    with open(annotation_path) as f:
        return len(f.readlines()) - 1
