from lmfit import Parameters, minimize, report_fit
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils import *

guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#001.json")

positions = np.array([snapshot.position for snapshot in snapshots])