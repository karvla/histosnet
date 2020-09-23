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

memory = Memory("./cache", verbose=0)

class Dataset:
    def __init__(self, path : Path):
        self.path = path
        self.image_dir = path / 'images'
        self.ids = np.array([Path(f).stem for f in os.listdir(self.image_dir)])
        self.image_shape = self.load_image(self.ids[0]).shape
    
    @property
    def size(self):
        return len(self.ids)

    def file_name(self, image_id):
        return str(self.image_dir / f"{image_id}.png")
    

    def load_image(self, image_id):
        return imread(self.image_dir / f"{image_id}.png")


    def make_split(self, factor = 0.8):
        np.random.seed(0)
        train_index = np.random.rand(self.size) < 0.8

        train_set = copy(self)
        train_set.ids = self.ids[train_index]

        test_set = copy(self)
        test_set.ids = self.ids[~train_index]
        return train_set, test_set

    def get_annotation(self, image_id):
        pass
            
        
 
class Monuseg(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / 'data/monuseg/')
        
    def get_annotation(self, patient_id : str):
        return utils.get_annotation_monuseg(patient_id)

class Swebcg(Dataset):
    def __init__(self):
        super().__init__(Path(__file__).parent.parent / 'data/swebcg/')

        with open(self.path / "annotations.json") as f:
            self.annotations = json.load(f)

    def get_annotation(self, patient_id : str):
        return [cell["vertices"] for cell in self.annotations[patient_id]]



if __name__ == "__main__":        
    monuseg = Monuseg() 
    pid = monuseg.ids[0]
    monuseg.get_annotation(pid)
    swebcg = Swebcg()
    print(swebcg.get_annotation(swebcg.ids[0]))
        
