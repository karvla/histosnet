import xml.etree.ElementTree as ET
from skimage.draw import polygon
from PIL import Image, ImageDraw
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from pathlib import Path
from tqdm import tqdm
from joblib import Memory
import numpy as np
import shapely.geometry as geo
import itertools as it

memory = Memory("./cache", verbose=0)
imdir = Path(__file__).parent.parent / "data/MoNuSeg Training Data/Tissue Images"
annotation_dir = Path(__file__).parent.parent / "data/MoNuSeg Training Data/Annotations"


@memory.cache
def get_mask(patient_id, shape=(1000, 1000)):
    return generate_binary_mask(get_annotation(patient_id), shape)


@memory.cache
def get_annotation(patient_id):
    return parse_xml_annotation_file(annotation_dir / f"{patient_id}.xml")


@memory.cache
def get_weight_map(patient_id):
    return unet_weight_map(get_mask(patient_id))
    #return generate_weight_map(
    #    patient_id, load_image(patient_id), get_annotation(patient_id), get_mask(patient_id)
    #)

# Stolen from https://stackoverflow.com/a/53179982
def unet_weight_map(mask, wc=None, w0 = 10, sigma = 5):

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
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(mask)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(mask)
    if wc:
        class_weights = np.zeros_like(mask)
        for k, v in wc.items():
            class_weights[mask == k] = v
        w = w + class_weights
    return w

def generate_weight_map(pid, im, annots, mask):
    print(f"Starting processing of {pid}")
    dims = im.shape[:2]
    weight_map = np.zeros(dims)

    if len(annots) >= 3:
        polygons = list(map(geo.Polygon, annots))
    else:
        polygons = []
    # loop over all points in image
    for x, y in it.product(range(dims[0]), range(dims[1])):
        # continue if point is inside a cell
        if mask[x, y]:
            weight_map[x, y] = 0
        else:
            point = geo.Point(x, y)
            d1, d2 = 10 ** 8, 10 ** 8
            for poly in polygons:
                distance = poly.distance(point)
                if distance < d1:
                    d2 = d1
                    d1 = distance
                elif distance < d2:
                    d2 = distance
            weight_map[int(point.y), int(point.x)] = (d1 + d2) ** 2

    return dsmap_to_weight_ma(weight_map)


def dsmap_to_weight_map(dsmap, w0=10, sigma=5):
    """
    Converts a dsmap to a weight map. A dsmap is a 2D array with the value of
    (d1 + d2)^2 for each pixel in an image, where d1 and d2 are the distances to the two cells
    in the image closest to that pixel. Returns the array as weight map following
    the same algorithm as the original UNET paper. I.e.
    w0 * np.exp(-dsmap/(2*sigma**2))
    """

    return w0 * np.exp(-dsmap / (2 * sigma ** 2))


def xml_annotations_to_dict(filepaths: list):
    """
    Produces an annotation dictionary from a list of paths to .xml files
    formatted in the monuseg format.
    """
    print("Parsing .xml files to dict...")
    annotations = dict()
    for fp in tqdm(map(Path, filepaths), total=len(filepaths)):
        annots = [{"vertices": a} for a in parse_xml_annotation_file(fp)]
        annotations[fp.stem] = annots
    return annotations


def discard_points(annotations: dict):
    """
    Discards all annotations consisting of only one vertex.

    Args:
        annotations: Dict with ids as keys and lists of lists of vertices as values.

    Returns:
        dict: Annotation dict with same sturcture but with no single point annotations.
    """
    print("Discarding points form annotations dict...")
    for patient, annots in tqdm(annotations.items()):
        for ind, annot in enumerate(annots):
            if len(annot["vertices"]) <= 2:
                annots.pop(ind)
            else:
                annot["vertices"] = list(map(tuple, annot["vertices"]))
    return annotations


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
        try:
            ImageDraw.Draw(img).polygon(annotation["vertices"], outline=1, fill=1)
        except:
            print(annotation)
    return np.array(img)


def generate_perimeter_mask(annotations, shape):
    img = Image.new("L", shape, 0)
    for annotation in annotations:
        ImageDraw.Draw(img).polygon(annotation["vertices"], outline=1, fill=0)
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


def load_image(patient_id):
    return imread(Path(imdir, patient_id + ".png"))


def bounding_box(vertices):
    x, y = zip(*vertices)
    return (min(x), min(y)), (max(x), max(y))
