import xml.etree.ElementTree as ET
import random
from skimage.draw import polygon
from PIL import Image, ImageDraw
from skimage.io import imread, imshow
from skimage.transform import rescale
from skimage.measure import label
from skimage.morphology import binary_erosion, diamond, remove_small_objects
from scipy.ndimage.morphology import distance_transform_edt
from pathlib import Path
from tqdm import tqdm
from joblib import Memory
import numpy as np
import shapely.geometry as geo
import itertools as it
from keras.utils.np_utils import to_categorical
import json
import dataset
from config import Config

c = Config()

memory = Memory("./cache", verbose=0)


@memory.cache
def get_mask(patient_id, shape=(1000, 1000)):
    return generate_binary_mask(get_annotation_monuseg(patient_id), shape)


@memory.cache
def get_annotation_monuseg(patient_id):
    annotation_dir = Path(__file__).parent.parent / "data/monuseg/annotations"
    return parse_xml_annotation_file(annotation_dir / f"{patient_id}.xml")


def get_annotation_swebcg(patient_id):
    annotations = Path(__file__).parent.parent / "data/swebcg/annotations.json"
    return parse_annotations(json.load(annotation)[patient_id])


@memory.cache
def get_weight_map(patient_id):
    return unet_weight_map(get_mask(patient_id))


@memory.cache
def get_weight_map_bns(image_id):
    mask = dataset.Bns().get_mask(image_id) > 0
    return unet_weight_map(mask, mask.shape[0])


def erode_mask(mask):
    return binary_erosion(mask, diamond(2))


def get_boundary_mask(mask):
    """ Returns mask where 0 is background, 1 in inside and 2 is boundary """
    return 2 * mask - erode_mask(mask)


# based on https://stackoverflow.com/a/53179982
def unet_weight_map(mask, win_size=100, w0=10, sigma=5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    win_size: int
        Size of window for calculating the weight mask in chunks.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    mask = erode_mask(mask)  # only use inside-values
    w = np.zeros_like(mask, dtype=np.float32)
    win_shape = (mask.shape[0], win_size)
    for j in range(0, mask.shape[1], win_size):
        labels = label(mask[:, j : j + win_size])
        no_labels = labels == 0
        label_ids = sorted(np.unique(labels))[1:]
        if len(label_ids) > 1:

            distances = np.zeros((win_shape + (len(label_ids),)))

            for i, label_id in enumerate(label_ids):
                distances[:, :, i] = distance_transform_edt(labels != label_id)

            label_ids = None
            distances = np.sort(distances, axis=2)
            w[:, j : j + win_size] = (
                w0
                * np.exp(
                    -1 / 2 * ((distances[:, :, 0] + distances[:, :, 1]) / sigma) ** 2
                )
                * no_labels
            )

    return w + mask


def parse_xml_annotation_file(filepath):
    """
    Parse an .xml file with the MoNuSeg annotation style annotations.

    Args:
        filepath: The path to the .xml file to parse.

    Returns:
        list: A list of annotations. The annotations are represented
              in the form of lists of (x,y) where x and y are the
              coordinates of a vertex of the polygon representing the
              annotation
    """

    tree = ET.parse(filepath)
    annotations = []

    for region in tree.find("Annotation").find("Regions").findall("Region"):
        vertices = []
        for vertex in region.find("Vertices").findall("Vertex"):
            vertices.append(
                (round(float(vertex.attrib["X"])), round(float(vertex.attrib["Y"])))
            )
        annotations.append(vertices)

    return annotations


def parse_annotations(annotations):
    for patient, annots in annotations.items():
        for ind, annot in enumerate(annots):
            if len(annot["vertices"]) <= 2:
                annots.pop(ind)
            else:
                annot["vertices"] = list(map(tuple, annot["vertices"]))
    return annotations


def generate_mask(annotations, shape):
    img = Image.new("L", shape, 0)
    for annotation in annotations:
        ImageDraw.Draw(img).polygon(annotation, outline=0, fill=1)
    return np.array(img)


def generate_perimeter_mask(annotations, shape):
    img = Image.new("L", shape, 0)
    for annotation in annotations:
        ImageDraw.Draw(img).polygon(annotation, outline=1, fill=0)
    return np.array(img)


def generate_binary_mask(annotations, shape):
    return generate_mask(annotations, shape) > 0


def draw_annotations(image, annotation, copy=False):
    if copy:
        image = image.copy()

    impath = Path(imdir, patient + ".png")
    annotpath = Path(imdir, patient + ".xml")

    im = Image.open(impath)
    shape = np.asarray(im).shape[:2]

    for annotation in annotations:
        draw_polygon(image, annotation)

    return image


def bounding_box(vertices):
    x, y = zip(*vertices)
    return (min(x), min(y)), (max(x), max(y))


def fix_wmap_shape(wmap, target_shape):
    wmap = wmap.ravel()

    #  weight map should only be applied to inside-values
    ones = np.ones(wmap.shape)
    wmap = np.array((ones, wmap, ones)).T
    wmap = wmap.reshape(target_shape)
    return wmap
