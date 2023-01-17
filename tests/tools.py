import datetime
import os
import pathlib
import pickle
from typing import List, Tuple, Callable
import shutil

import numpy as np

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop import CodeLabeler

def event(date: Tuple, code, value, visit_id=None):
    """A terser way to create a Piton Event."""
    hour, minute, seconds = 0, 0, 0
    if len(date) == 3:
        year, month, day = date
    elif len(date) == 4:
        year, month, day, hour = date
    elif len(date) == 5:
        year, month, day, hour, minute = date
    elif len(date) == 6:
        year, month, day, hour, minute, seconds = date
    else:
        raise ValueError(f"Invalid date: {date}")
    return piton.Event(
        start=datetime.datetime(
            year,
            month,
            day,
            hour,
            minute,
            seconds,
        ),
        code=code,
        value=value,
        visit_id=visit_id,
    )
    
def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)
        
def run_test_locally(str_path: str, test_func: Callable):
    """Run test locally the way Github Actions does (in a temporary directory `tmp_path`)."""
    tmp_path = pathlib.Path(str_path)
    shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    test_func(pathlib.Path(tmp_path))