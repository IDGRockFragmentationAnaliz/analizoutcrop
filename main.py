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
from get_log_density_data import get_log_density_data


def main():
    data_folder = Path("D://1.ToSaver//data//outcrop_data")
    
    with open("./data/config.json") as file:
        config = json.load(file)
    
    for i, image_folder in enumerate(data_folder.iterdir()):
        print(image_folder.name)
        image_data, _ = ImageData.load(image_folder)
        #
        s = get_areas(image_data)
        data = get_ks_test_data(s, config[image_folder.name])
        print("lognorm:")
        print_test_data(data["test_data"]["lognorm"])
        print("paretoexp:")
        print_test_data(data["test_data"]["paretoexp"])
        print("weibull:")
        print_test_data(data["test_data"]["weibull"])
        
        headers = []
        distributions = ['lognorm', 'paretoexp', 'weibull']
        metrics = ['Hypothesis', 'd', 'p-value', 'theta1', 'theta2']
        
        
def print_test_data(test_data):
    ks_test = test_data['ks_test']
    d = test_data['d']
    p_value = test_data['p_value']
    theta = test_data["theta"]
    print(
        f"hypothesis: {ks_test}, "
        f"d: {d:.3f}, "
        f"p_value: {p_value:.3f}",
        f"theta: {theta[0]:.3f}, {theta[1]:.3f}",
    )


def density_culcs(data_folder, config):
    out_data_density = {}
    for i, image_folder in enumerate(data_folder.iterdir()):
        print(image_folder.name)
        image_data, _ = ImageData.load(image_folder)
        #
        s = get_areas(image_data)
        # density evaluate
        data_density = get_density_data(s, config[image_folder.name])
        out_data_density[image_folder.name] = data_density
        
    with open("./data/outcrops_densities.json", 'w+') as json_file:
        json.dump(out_data_density, json_file, indent=4)

    
def get_ks_test_data(s, config):
    pix2m = (config["m"] / config["pix"])
    s = np.delete(s, np.argmax(s))
    
    areas = s * (pix2m**2)

    xmin = np.min(areas)
    xmax = np.max(areas)

    models = {"lognorm": lognorm, "paretoexp": paretoexp, "weibull": weibull}
    tests = {name: DistributionTest(areas, model) for name, model in models.items()}
    
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
    }
    return data

    
def plot_statistic_data(data):
    fig = plt.figure(figsize=(12, 4))
    axs = [fig.add_subplot(1, 1, 1)]
    x = data["x"]
    y_min = np.array(data["test_data"]["lognorm"]["cdf_min"])
    y_max = np.array(data["test_data"]["lognorm"]["cdf_max"])
    print("lognorm: ", np.mean(y_max - y_min)/2)
    axs[0].fill_between(x, y_min, y_max, color="red", linewidth=2, alpha=0.5)
    y_min = np.array(data["test_data"]["paretoexp"]["cdf_min"])
    y_max = np.array(data["test_data"]["paretoexp"]["cdf_max"])
    print("paretoexp: ", np.mean(y_max - y_min)/2)
    axs[0].fill_between(x, y_min, y_max, color="green", linewidth=2, alpha=0.5)
    y_min = np.array(data["test_data"]["weibull"]["cdf_min"])
    y_max = np.array(data["test_data"]["weibull"]["cdf_max"])
    print("weibull: ", np.mean(y_max - y_min)/2)
    axs[0].fill_between(x, y_min, y_max, color="blue", linewidth=2, alpha=0.5)
    axs[0].set_xscale('log')
    x = data["ecdf"]["values"]
    y = data["ecdf"]["freqs"]
    axs[0].plot(x, y, color="black")
    plt.show()


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
