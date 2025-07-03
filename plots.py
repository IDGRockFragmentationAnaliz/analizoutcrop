from pathlib import Path

import numpy as np
import cv2
import scipy as sp
import json
import matplotlib.pyplot as plt


from rocknetmanager.tools.image_data import ImageData
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from pyrockstats.bootstrap.ks_statistics import get_ks_distribution
from pyrockstats.bootstrap.ks_statistics import get_confidence_value
from pyrockstats.empirical import ecdf
from distrebution_test import DistributionTest


def main():
    with open("./data/outcrops_tests.json") as file:
        data = json.load(file)
    data = data["IMGP3286"]
    bins = data["bins"]
    hist = data["hist"]

if __name__ == "__main__":
    main()
