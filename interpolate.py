from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Dict, Iterable, Optional, Tuple

from fetch_iso import IsochroneFetcher
from find_mag import PhotometryMerger
from read_spot_models import SPOT

phot_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv'
dist_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv'
targets = PhotometryMerger.join_photometry_and_distances(phot_csv, dist_csv)
dfs = SPOT("isochrones/SPOTS/isos/f000.isoc").read_iso_file()


def get_spot_info(dfs):
    for df in dfs:







