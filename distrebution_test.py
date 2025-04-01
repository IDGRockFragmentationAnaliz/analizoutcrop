from pathlib import Path

import numpy as np

from rocknetmanager.tools.shape_load import shape_load
from rocknetmanager.tools.image_data import ImageData
import cv2
import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
import scipy as sp

from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from pyrockstats.bootstrap.ks_statistics import get_ks_distribution
from pyrockstats.bootstrap.ks_statistics import get_confidence_value
from pyrockstats.empirical import ecdf
import json


class DistributionTest:
    def __init__(self, areas, model):
        self.xmin = np.min(areas)
        self.xmax = np.max(areas)
        self.areas = areas
        self.model = model
        self.ks = get_ks_distribution(areas, model, n_ks=500)
        self.theta = self.model.fit(areas, xmin=self.xmin, xmax=self.xmax)
        self.dist = self.model(*self.theta, xmin=self.xmin, xmax=self.xmax)
        self.confidence_value = None
        self.alpha = None
        self.hypothesis = None

    def get_confidence_value(self, alpha):
        if self.alpha is not None and alpha == self.alpha:
            return self.confidence_value
        self.alpha = alpha
        self.confidence_value = get_confidence_value(self.ks, significance=alpha)
        return self.confidence_value

    def model_cdf(self, x):
        return self.dist.cdf(x, xmin=self.xmin, xmax=self.xmax)

    def ks_test(self, alpha, e_values = None, e_cdf = None):
        if e_values is None or e_cdf is None:
            e_values, e_cdf = ecdf(self.areas)
        confidence_value = self.get_confidence_value(alpha)
        cdf_min = self.model_cdf(e_values) - confidence_value
        cdf_max = self.model_cdf(e_values) + confidence_value
        self.hypothesis = np.all(cdf_min < e_cdf) and np.all(cdf_max > e_cdf)
        return self.hypothesis

    def get_data(self, x, alpha):
        confidence_value = self.get_confidence_value(alpha)
        cdf = self.model_cdf(x)
        cdf_min = cdf - confidence_value
        cdf_max = cdf + confidence_value
        data = {
            "cdf": cdf.tolist(),
            "cdf_min": cdf_min.tolist(),
            "cdf_max": cdf_max.tolist(),
            "ks_test": str(self.ks_test(alpha))
        }
        return data