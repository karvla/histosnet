from skimage.transform import resize
from skimage import morphology
from skimage import measure
from scipy.ndimage import find_objects
from joblib import Memory
from pathlib import Path
from time import time
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, True)

from skimage.io import imread, imsave
import click
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.utils import Sequence
from keras.models import Model, load_model
from keras.layers import Input, multiply
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from weighted_loss_unet import make_weighted_loss_unet, my_loss
import config
import pickle

c = config.Config()
memory = Memory("./cache", verbose=False)


def iou():
    pass


def brightness_mean(im):
    return np.nanmean(np.ravel(np.where(im != 0, im, np.nan)))


def brightness_var(im):
    return np.var(np.ravel(im))


def avg_brightness(ids, masks, obj, image_dict):
    avg = []
    var = []

    for imid, mask, obj in zip(ids, masks, obj):
        img = image_dict[imid][obj]
        mask = mask[: img.shape[0], : img.shape[1]]
        img = img[: mask.shape[0], : mask.shape[1], :]
        img = img * np.dstack((mask, mask, mask))
        m = brightness_mean(img)
        v = brightness_var(img)

        if m is None or v is None:
            avg.append(0)
            var.append(0)
        else:
            avg.append(m)
            var.append(v)
    return avg, var


def _reshape_to_batch(patch):
    n = int(np.sqrt(c.BATCH_SIZE))
    batch = []
    for col in np.array_split(patch, n, axis=0):
        for b in np.array_split(col, n, axis=1):
            batch.append(b)
    return np.asarray(batch)


def _reshape_to_patch(batch):
    n = int(np.sqrt(c.BATCH_SIZE))
    patch = np.zeros((n * c.HEIGHT, n * c.WIDTH))
    i = 0
    for y in range(0, n * c.HEIGHT, c.HEIGHT):
        for x in range(0, n * c.WIDTH, c.WIDTH):
            patch[y : y + c.HEIGHT, x : x + c.WIDTH] = batch[i, ...]
            i += 1
    return patch


def post_processing(prediction, p=0.08):
    prediction = prediction > p
    prediction = morphology.dilation(prediction, morphology.square(2))
    return prediction


@memory.cache
def make_pred(img_path, model_name, stride):
    model = keras.models.load_model(
        c.MODEL_DIR / f"unet/{model_name}.h5",
        custom_objects={"my_loss": my_loss, "iou": iou},
        compile=False,
    )
    img = imread(img_path)

    b_width = int(np.sqrt(c.BATCH_SIZE))
    pred = np.zeros(img.shape[0:2])
    norm_mat = np.ones_like(pred)
    for y in range(0, pred.shape[1], stride):
        for x in range(0, pred.shape[0], stride):
            t0 = time()
            img_patch = img[y : y + b_width * c.HEIGHT, x : x + b_width * c.WIDTH]
            batch_patch = np.zeros((b_width * c.HEIGHT, b_width * c.HEIGHT, c.CHANNELS))
            batch_patch[: img_patch.shape[0], : img_patch.shape[1]] = img_patch
            batch = _reshape_to_batch(batch_patch)
            pred_batch = model.predict(batch)[..., 1]
            K.clear_session()
            pred_patch = _reshape_to_patch(pred_batch)
            pred[y : y + b_width * c.HEIGHT, x : x + b_width * c.WIDTH] += pred_patch[
                : img_patch.shape[0], : img_patch.shape[1]
            ]
            norm_mat[y : y + b_width * c.HEIGHT, x : x + b_width * c.WIDTH] += 1

    pred = pred / norm_mat
    return pred


def get_objects(mask):
    all_labels = measure.label(mask, background=0)
    return find_objects(all_labels)


def make_pred_dataframe(key, mask, image):
    """Takes an image id  and a mask
    and returns a dataframe with all cells"""

    classes = ["tumor", "immunes cells"]
    with open(c.MODEL_DIR / "cell_classifier.pickle", "rb") as f:
        features, model = pickle.load(f)
    d = []
    for obj in get_objects(mask):
        m = mask[obj]
        d.append({"image_id": key, "obj": obj, "mask": m})

    df_pred = pd.DataFrame(d)
    df_pred["avg_brightness"], df_pred["var_brightness"] = avg_brightness(
        df_pred["image_id"], df_pred["mask"], df_pred["obj"], {key: image}
    )
    df_pred["size"] = df_pred["mask"].apply(np.sum)
    df_pred["class"] = model.predict(df_pred[features].iloc[:])
    df_pred["class_name"] = df_pred["class"].apply(lambda x: classes[x])
    return df_pred


def cell_locations(df: pd.DataFrame):
    return (
        df["obj"]
        .apply(lambda x: ((x[0].start + x[0].stop) / 2, (x[1].start + x[1].stop) / 2))
        .values
    )


def make_patient_dataframe(df_pred):
    """ Returns a dataframe with one row for every image. """
    d = []
    for image_id in df_pred["image_id"].unique():
        df_imid = df_pred[df_pred["image_id"] == image_id]
        d.append(
            {
                "image_id": image_id,
                "n_tumor": len(df_imid[df_imid["class_name"] == "tumor"].index),
                "n_immune": len(df_imid[df_imid["class_name"] == "immune cells"].index),
                "tumor_loc": cell_locations(df_imid[df_imid["class_name"] == "tumor"]),
                "immune_loc": cell_locations(
                    df_imid[df_imid["class_name"] == "immune cells"]
                ),
                "tumor_area": df_imid[df_imid["class_name"] == "tumor"]["size"].sum(),
                "immune_area": df_imid[df_imid["class_name"] == "immune cells"][
                    "size"
                ].sum(),
                "tumor_obj" : df_imid[df_imid["class_name"] == "tumor"]["obj"],
                "immune_obj" : df_imid[df_imid["class_name"] == "immune cells"]["obj"],
            }
        )
    return pd.DataFrame(d)


@click.command()
@click.option("--source", "-s", type=click.Path())
@click.option("--destination", "-d", type=click.Path())
@click.option("--model_name", "-m")
def predict_folder(source, destination, model_name):

    model = keras.models.load_model(
        c.MODEL_DIR / f"unet/{model_name}.h5",
        custom_objects={"my_loss": my_loss, "iou": iou},
        compile=False,
    )

    for name in tqdm(os.listdir(source)):
        path = os.path.join(source, name)
        pred = make_pred(path, model)
        imsave(destination + "pred" + name, pred)


if __name__ == "__main__":
    predict_folder()
