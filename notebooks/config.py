from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    WIDTH = 256
    HEIGHT = 256
    CHANNELS = 3
    SHAPE = (WIDTH, HEIGHT, CHANNELS)

    BATCH_SIZE = 9
    BATCH_SHAPE = (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
    EPOCHS = 400

    ROOT_DIR = Path(__file__).parent.parent 
    MODEL_DIR = ROOT_DIR / 'models'

