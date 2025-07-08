from pathlib import Path
import math
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
        self.ks = get_ks_distribution(areas, model, n_ks=1000)
        self.theta = self.model.fit(areas, xmin=self.xmin, xmax=self.xmax)
        self.dist = self.model(*self.theta, xmin=self.xmin, xmax=self.xmax)
        self.confidence_value = None
        self.alpha = None
        self.hypothesis = None
        self.d = None
        self.e_cdf = None
        self.e_values = None
        self.p_value = None

    def get_confidence(self, alpha):
        if self.alpha is not None and alpha == self.alpha:
            return self.confidence_value
        self.alpha = alpha
        self.confidence_value = get_confidence_value(self.ks, significance=alpha)
        return self.confidence_value

    def model_cdf(self, x):
        return self.dist.cdf(x, xmin=self.xmin, xmax=self.xmax)
    
    def get_ecdf(self):
        if self.e_values is not None and self.e_cdf is not None:
            return self.e_values, self.e_cdf
        self.e_values, self.e_cdf = ecdf(self.areas)
        return self.e_values, self.e_cdf
    
    def get_ks_norm(self):
        if self.d is not None:
            return self.d
        e_values, e_cdf = self.get_ecdf()
        self.d = np.max(np.abs(self.model_cdf(e_values) - e_cdf))
        return self.d
    
    def ks_test(self, alpha):
        if self.hypothesis is not None and self.alpha == alpha:
            return self.hypothesis
        self.hypothesis = self.get_ks_norm() <= self.get_confidence(alpha)
        return self.hypothesis
    
    def get_p_value(self):
        if self.p_value is not None:
            return self.p_value
        d = self.get_ks_norm()
        e_values, e_cdf = ecdf(self.ks)
        idx = np.searchsorted(e_values, d, side='right')
        if idx == 0:
            p_value = 1.0  # Если D меньше всех значений в self.ks
        else:
            p_value = 1.0 - e_cdf[idx - 1]
        self.p_value = p_value
        return self.p_value
    
    def get_data(self, x, alpha):
        confidence_value = self.get_confidence(alpha)
        cdf = self.model_cdf(x)
        cdf_min = cdf - confidence_value
        cdf_max = cdf + confidence_value
        hypothesis = self.ks_test(alpha)
        d = self.get_ks_norm()
        p_value = self.get_p_value()
        theta = self.theta
        data = {
            "cdf": cdf.tolist(),
            "cdf_min": cdf_min.tolist(),
            "cdf_max": cdf_max.tolist(),
            "ks_test": str(hypothesis),
            "d": d,
            "p_value": p_value,
            "theta": theta
        }
        return data
    