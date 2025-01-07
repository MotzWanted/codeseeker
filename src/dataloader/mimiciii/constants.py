import pathlib
from dataloader.mimiciii import mimiciii_full
from dataloader.mimiciii import mimiciii_50

MIMIC_III_CLEAN_PATH = str(pathlib.Path(mimiciii_full.__file__))
MIMIC_III_50_PATH = str(pathlib.Path(mimiciii_50.__file__))
