from time import time
import os
import sys
import shutil
import click
from pathlib import Path
import config
from openslide import OpenSlide
import pandas as pd
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from skimage.io import imread
from skimage.transform import rescale
import random
import numpy as np
import keras
from weighted_loss_unet import *
from predict import (
    predict_image,
    make_pred_dataframe,
    make_patient_dataframe,
    post_processing,
    predict_path
)

c = config.Config()


def safe_save(df: pd.DataFrame, location: Path):
    tmp = Path(__file__).parent / ".tmp.pickle"
    df.reset_index(drop=True).to_feather(tmp)
    shutil.copy(tmp, location)


def mask(img):
    img_mean = np.mean(img, axis=-1) / 255
    mask = np.logical_and(0.1 < img_mean, img_mean < 0.9)
    mask = remove_small_objects(mask, np.sum(mask) * 0.1)
    return mask


def tissue_positions(slide: OpenSlide):
    thumbnail = slide.get_thumbnail((1000, 1000))
    tissue = mask(thumbnail)
    scale_factor = max(slide.dimensions) / max(thumbnail.size)
    coords = np.where(tissue)
    coords = [(c * scale_factor).astype(np.int) for c in coords]
    coords = list(zip(*coords))
    return coords


def slide_patches(slide: OpenSlide, n=10, width=1024):
    coords = tissue_positions(slide)
    for y, x in coords[:: int(len(coords) / n)]:
        y, x = y - int(width / 2), x - int(width / 2)
        yield np.array(slide.read_region((x, y), 0, (width, width)))[..., :3]


@click.command()
@click.option(
    "--destination",
    "-d",
    type=click.Path(file_okay=False, dir_okay=True),
    default=Path(__file__).parent,
    help="Destination folder.",
)
@click.option(
    "--model_name", "-m", default="unet_quip_10000", help="Name of segmentation model."
)
@click.option("--cutoff", "-c", default=0.05, help="Cutoff for the segmentation model.")
@click.option("--min_size", "-s", default=5, help="Smallest size in pixels for a cell.")
@click.option(
    "--scale",
    "-r",
    default=1.0,
    help="Target scale when rescaling image during prediction of segmentation map. Might improve result if the images are captures a different magnification than x40.",
)
@click.option(
    "--n_samples",
    "-n",
    default=200,
    help="The number samples segmended from the whole slide image.",
)
@click.option(
    "--stride",
    default= 256,
    help="Stride in pixels for segmentation prediction.",
)
@click.argument("source", type=click.Path())
@click.argument("image_type", type=click.Choice(["WSI", "TMA"]))
def main(**kwargs):
    image_type = kwargs["image_type"]

    destination_file = Path(kwargs["destination"]) / _file_name(**kwargs)
    started = os.path.exists(destination_file)

    if started:
        print("Resuming..")
        with open(destination_file) as f:
            df_pat = pd.read_feather(destination_file)
    else:
        print("Starting feature extration..")
        df_pat = pd.DataFrame()

    n_total = len(os.listdir(kwargs["source"]))
    df_pat = pd.DataFrame()
    for n, image_path in enumerate(Path(kwargs["source"]).iterdir()):
        print(f"image {n} of {n_total}")

        image_id = Path(image_path).name
        if started and image_id in df_pat["image_id"].unique():
            continue

        if image_type == "WSI":
            df_pat = pd.concat([df_pat, features_wsi(image_path, kwargs)])
        elif image_type == "TMA":
            df_pat = pd.concat([df_pat, features_tma(image_path, kwargs)])
        else:
            raise ValueError

        safe_save(df_pat, destination_file)
    print("...done!")


def features_wsi(image_path, kwargs):
    n_samples = kwargs["n_samples"]
    scale = kwargs["scale"]
    stride = kwargs["stride"]
    cutoff = kwargs["cutoff"]
    min_size = kwargs["min_size"]
    model_name = kwargs["model_name"]

    model = keras.models.load_model(
        c.MODEL_DIR / f"unet/{model_name}.h5",
        custom_objects={"my_loss": my_loss},
        compile=False,
    )

    slide = OpenSlide(str(image_path))
    df_pat = pd.DataFrame()
    for img in tqdm(
        slide_patches(slide, n_samples),
        total=n_samples,
    ):

        if scale != 1.0:
            img = rescale(img, scale, preserve_range=True, multichannel=True)
            pred = predict_image(img, model, stride)
            pred = rescale(pred, 1 / scale, preserve_range=True, multichannel=False)
        else:
            pred = predict_image(img, model, stride)

        mask = post_processing(pred, cutoff, min_size)

        if np.sum(mask):
            df_cells = make_pred_dataframe(image_path.name, mask, img)
            df_pat = pd.concat([df_pat, make_patient_dataframe(df_cells)])

    return df_pat


def features_tma(image_path, kwargs):
    scale = kwargs["scale"]
    stride = kwargs["stride"]
    cutoff = kwargs["cutoff"]
    min_size = kwargs["min_size"]
    model_name = kwargs["model_name"]

    pred = predict_path(image_path, model_name, stride, scale)
    mask = post_processing(pred, cutoff, min_size)
    img = imread(image_path)

    if np.sum(mask):
        df_cells = make_pred_dataframe(image_path.name, mask, img)
        df_pat = make_patient_dataframe(df_cells)
        return df_pat
    else:
        return pd.DataFrame()

def _file_name(**kwargs):
    if kwargs["image_type"] == "WSI":
        return f"wsi_{kwargs['model_name']}_{kwargs['cutoff']}_{kwargs['min_size']}_{kwargs['scale']}_{kwargs['n_samples']}.feather"

    elif kwargs["image_type"] == "TMA":
        return f"tma_{kwargs['model_name']}_{kwargs['cutoff']}_{kwargs['min_size']}_{kwargs['scale']}.feather"
    else:
        raise ValueError



if __name__ == "__main__":
    main()
