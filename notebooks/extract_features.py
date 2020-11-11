from weighted_loss_unet import *
from time import time
import os
import sys
import shutil
import click
from pathlib import Path
import keras
import config
from openslide import OpenSlide
import pandas as pd
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from skimage.transform import rescale
import random
import numpy as np
from predict import (
    predict_image,
    make_pred_dataframe,
    make_patient_dataframe,
    post_processing,
)

c = config.Config()


def safe_save(df: pd.DataFrame, location: Path):
    tmp = Path(__file__).parent / ".tmp.pickle"
    df_pat.reset_index().to_feather(tmp)
    shutil.copy(tmp, location)


def tissue_positions(slide: OpenSlide):
    thumbnail = slide.get_thumbnail((100, 100))
    tissue = np.mean(thumbnail, axis=-1) / 255 < 0.9
    tissue = remove_small_objects(tissue, 200)
    scale_factor = slide.dimensions[0] / thumbnail.size[0]
    coords = np.where(tissue)
    coords = [(c * scale_factor).astype(np.int) for c in coords]
    coords = list(zip(*coords))
    return coords


def slide_patches(slide: OpenSlide, n=10, width=1000):
    coords = tissue_positions(slide)
    for y, x in random.choices(coords, k=n):
        y, x = y - int(width / 2), x - int(width / 2)
        yield np.array(slide.read_region((x, y), 0, (width, width)))[..., :3]


def iou():
    pass


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
    default=int(3 * c.HEIGHT / 2),
    help="Stride in pixels for segmentation prediction.",
)
@click.argument("source", type=click.Path())
def main(
    destination,
    model_name,
    cutoff,
    min_size,
    scale,
    n_samples,
    stride,
    source,
):

    destination_file = (
        Path(destination)
        / f"wsi_{model_name}_{cutoff}_{min_size}_{scale}_{n_samples}.feather"
    )

    model = keras.models.load_model(
        c.MODEL_DIR / f"unet/{model_name}.h5",
        custom_objects={"my_loss": my_loss, "iou": iou},
        compile=False,
    )

    started = os.path.exists(destination_file)

    if started:
        print("Resuming..")
        with open(destination_file) as f:
            df_pat = pd.read_feather(destination_file)
    else:
        print("Starting feature extration..")
        df_pat = pd.DataFrame()

    images = os.listdir(source)
    for n, image_name in enumerate(images):
        image_id = Path(image_name).stem
        if started and image_id in df_pat["image_id"].unique():
            continue

        slide = OpenSlide(os.path.join(source, image_name))
        for img in tqdm(
            slide_patches(slide, n_samples),
            total=n_samples,
            desc=f"image {n} of {len(images)}",
        ):

            if scale != 1.0:
                img = rescale(img, scale, preserve_range=True, multichannel=True)
                pred = predict_image(img, model, stride)
                pred = rescale(pred, 1 / scale, preserve_range=True, multichannel=False)
            else:
                pred = predict_image(img, model, stride)

            mask = post_processing(pred, cutoff, min_size)

            if np.sum(mask):
                try:
                    df_cells = make_pred_dataframe(image_id, mask, img)
                    df_pat = pd.concat([df_pat, make_patient_dataframe(df_cells)])
                except Exception as e:
                    print(e)
        safe_save(df_pat, destination_file)
    print("...done!")


if __name__ == "__main__":
    main()
