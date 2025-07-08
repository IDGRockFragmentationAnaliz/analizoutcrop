from contextlib import nullcontext
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


def get_log_density_data(s, config):
    # вычисление размера пикселя
    pix2m = (config["m"] / config["pix"])
    s = np.delete(s, np.argmax(s))
    xmin = np.min(s)
    xmax = np.max(s)

    # Начальное число бинов
    n_bins = 10
    min_bins = 7  # минимальное допустимое число бинов

    # Логарифмические бины
    hist = None
    bins = None
    while True:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins)
        hist, bins = np.histogram(s, bins=bins)
        if np.all(hist > 0):
            break
        elif n_bins > min_bins:
            n_bins -= 1
        else:
            break

    # маска для ненулевых значений гистограммы
    mask = hist > 0

    # Вычисляем плотность
    bin_widths = np.diff(bins)
    rho = np.log10(hist[mask]) - np.log10(bin_widths[mask] * np.sum(s)) - 4*np.log10(pix2m)

    # Средние точки бинов
    s_rho = (bins[:-1] + bins[1:]) / 2
    s_rho = np.log10(s_rho[mask]) + 2*np.log10(pix2m)

    # Преобразуем в список для JSON-сериализации
    data = {
        "s": s_rho.tolist(),
        "rho": rho.tolist(),
        "unit": "log m2"
    }
    return data
