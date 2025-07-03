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


def main():
    data_folder = Path("/media/koladik/HardDisk/data/OutcropData")
    out_data_density = {}
    with open("./data/config.json") as file:
        config = json.load(file)

    for i, image_folder in enumerate(data_folder.iterdir()):
        print(image_folder.name)
        image_data, _ = ImageData.load(image_folder)
        #
        s = get_areas(image_data)
        # density evaluate
        data_density = get_density_data(s, config[image_folder.name])
        out_data_density[image_folder.name] = data_density

    with open("./data/outcrops_tests.json", 'w+') as json_file:
        json.dump(out_data_density, json_file, indent=4)


def get_density_data(s, config):
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
    rho = np.log10(hist[mask]) - np.log10(bin_widths[mask] * np.sum(s)) + 4*np.log10(pix2m)

    # Средние точки бинов
    s_rho = (bins[:-1] + bins[1:]) / 2
    s_rho = np.log(s_rho[mask]) + 2*np.log10(pix2m)

    # Преобразуем в список для JSON-сериализации
    data = {
        "s": s_rho.tolist(),
        "rho": rho.tolist(),
        "unit": "log m2"
    }
    return data


def get_data(s, config):
    pix2m = (config["m"] / config["pix"])
    s = np.delete(s, np.argmax(s))

    name = image_folder.name

    pix2m2 = (config[name]["m"] / config[name]["pix"]) ** 2

    s = s[1:]
    areas = s * pix2m2

    xmin = np.min(areas)
    xmax = np.max(areas)

    models = {"lognorm": lognorm, "paretoexp": paretoexp, "weibull": weibull}
    tests = {name: DistributionTest(areas, model) for name, model in models.items()}

    ks_tests = {name: test.ks_test(0.05) for name, test in tests.items()}
    thetas = {name: (float(test.theta[0]), float(test.theta[1])) for name, test in tests.items()}
    values, e_freq = ecdf(areas)
    x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    alpha = 0.05
    data = {
        "x": x.tolist(),
        "xmin": xmin,
        "xmax": xmax,
        "alpha": alpha,
        "test_data": {name: test.get_data(x, alpha) for name, test in tests.items()},
        "ecdf": {"values": values.tolist(), "freqs": e_freq.tolist()},
        "theta": {name: tests[name].theta for name, test in tests.items()},
        "units": "m2"
    }
    return data


def get_areas(image_data: ImageData):
    label, _, _ = cv2.split(image_data.label)
    area, _, _ = cv2.split(255 - image_data.area)
    #
    segments = Segmentator(label)
    marks = segments.run()
    marks_image = segments.get_segment_image()
    cv2.imwrite("./data/image_3.png", marks_image)
    unique, s = np.unique(marks, return_counts=True)

    return s


if __name__ == "__main__":
    main()



# fig = plt.figure(figsize=(12, 4))
# axs = [fig.add_subplot(1, 1, 1)]
# axs[0].plot(x, tests["lognorm"].model_cdf(x))
# axs[0].plot(data["ecdf"]["values"], data["ecdf"]["freqs"], color="black", linestyle="--", label="2")
# axs[0].set_xscale('log')
# plt.show()