import pathlib
from ml_datasets.tanner_health import (
    tanner_health_inpatient,
    tanner_health_surgery,
    tanner_health_ed,
)

TANNER_INPATIENT_PATH = str(pathlib.Path(tanner_health_inpatient.__file__))
TANNER_SURGERY_PATH = str(pathlib.Path(tanner_health_surgery.__file__))
TANNER_ED_PATH = str(pathlib.Path(tanner_health_ed.__file__))
