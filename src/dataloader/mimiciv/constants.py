import pathlib

from dataloader.mimiciv import mimiciv, mimiciv_50

MIMIC_IV_PATH = str(pathlib.Path(mimiciv.__file__))
MIMIC_IV_50_PATH = str(pathlib.Path(mimiciv_50.__file__))
