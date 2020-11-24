from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    WIDTH = 512
    HEIGHT = WIDTH
    CHANNELS = 3
    SHAPE = (WIDTH, HEIGHT, CHANNELS)

    BATCH_SIZE = 1
    BATCH_SHAPE = (BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
    EPOCHS = 6000

    CLASS_WEIGHTS = [1.0, 1.0, 1.0]

    ROOT_DIR = Path(__file__).parent.parent
    MODEL_DIR = ROOT_DIR / "models"
