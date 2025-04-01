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


def get_areas(image_data: ImageData):
    label, _, _ = cv2.split(image_data.label)
    area, _, _ = cv2.split(255 - image_data.area)
    #
    segments = Segmentator(label)
    marks = segments.run()
    marks_image = segments.get_segment_image()
    cv2.imwrite("./data/image_3.png", marks_image)
    
    # fig = plt.figure(figsize=(12, 4))
    # axs = [fig.add_subplot(1, 2, 1),
    #        fig.add_subplot(1, 2, 2)]
    # axs[0].imshow(area_marks)
    # axs[1].imshow(image_data.image)
    # plt.show()
    
    #exit()
    unique, s = np.unique(marks, return_counts=True)
    
    return s



def main():
    data_folder = Path("D:/1.ToSaver/profileimages/photo_database_complited")
    data = {}
    with open("./data/config.json") as file:
        config = json.load(file)
    
    for i, image_folder in enumerate(data_folder.iterdir()):
        name = image_folder.name
        image_data, _ = ImageData.load(image_folder)
        s = get_areas(image_data)
        if i < 1:
            continue
        return
        s = s[1:]
        areas = s*pix2m2
    
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
            "theta": {name: tests[name].theta for name, test in tests.items()}
        }
        
        # fig = plt.figure(figsize=(12, 4))
        # axs = [fig.add_subplot(1, 1, 1)]
        # axs[0].plot(x, tests["lognorm"].model_cdf(x))
        # axs[0].plot(data["ecdf"]["values"], data["ecdf"]["freqs"], color="black", linestyle="--", label="2")
        # axs[0].set_xscale('log')
        # plt.show()
        # exit()
        out_data[name] = data
        # print(ks_tests)
        print(thetas)
        # exit()
        # s = np.sum(areas)
        # bins = np.logspace(x_min, x_max, 10)
        # hist, bins = np.histogram(areas, bins)
    exit()
    with open("./data/outcrops_tests.json", 'w+') as json_file:
        json.dump(out_data, json_file, indent=4)
    
    #     bins = bins * pix2m2
    #     hist = hist / (s * pix2m2)
    #
    #     data[name] = {
    #         "bins": list(bins),
    #         "density": list(hist),
    #         "units": "m2"
    #     }
    #
    # with open("data/section_hists.json", 'w') as json_file:
    #     json.dump(data, json_file, indent=4)

    exit()
    #sp.io.savemat("./Section_hists.mat", data)

        # fig = plt.figure(figsize=(16, 4))
        # ax = [fig.add_subplot(1, 1, 1)]
        # ax[0].imshow(image_data.image)
        # plt.show()
        # exit()



if __name__ == "__main__":
    main()

