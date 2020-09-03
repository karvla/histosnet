from pathlib import Path
import json
import numpy as np
import random

from PIL import Image
from PIL import Image, ImageDraw
from skimage.draw import polygon
from skimage.io import imread, imshow
from skimage.transform import resize

def load_json(filepath):
    with open(filepath) as f:
        annotations = json.load(f)
    return annotations


def parse_annotations(filepath):
    """
    
    """
    with open(filepath) as f:
        annotations = json.load(f)
        
    for patient, annots in annotations.items():
        for annot in annots:
            annot['vertices'] = list(map(tuple, annot['vertices']))
    return annotations


def generate_mask(annotations, shape):
    img = Image.new('L', shape, 0)
    for annotation in annotations:
        ImageDraw.Draw(img).polygon(annotation['vertices'], outline=1, fill=1)
    return np.array(img)


def generate_binary_mask(annotations, shape):
    return generate_mask(annotations, shape) > 0

    
def poly_to_mask(vertices, shape):
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
    return np.array(img)


def draw_polygon(image, vertices, fill=None):
    ImageDraw.Draw(image).polygon(vertices, outline=1, fill=fill)
    return image
    
    
def draw_annotations(image, annotation, copy=False):
    if copy:
        image = image.copy()
        
    impath = Path(imdir, patient + '.png')
    annotpath = Path(imdir, patient + '.xml')
    
    im = Image.open(impath)
    shape = np.asarray(im).shape[:2]
    
    for annotation in annotations:
        draw_polygon(image, annotation)
        
    return image
    

def load_image(patient_id):
    return imread(Path(imdir, patient_id + '.png'))
    
    
def bounding_box(vertices):
    x, y = zip(*vertices)
    return (min(x), min(y)), (max(x), max(y))


def val_split(ids, val_ratio=0.2):
    ids = list(ids)
    random.shuffle(ids)
    k = round(len(ids)*val_ratio)
    return ids[k:], ids[:k]
