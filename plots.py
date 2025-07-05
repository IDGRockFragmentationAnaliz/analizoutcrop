from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt


def main():
    with open(".//data//outcrops_densities.json") as file:
        data = json.load(file)

    for name in data:
        print(name)
        _data = data[name]
        _data["x"] = _data["s"]
        _data["y"] = _data["rho"]


    fig = plt.figure(figsize=(12, 4))
    axs = [fig.add_subplot(1, 1, 1)]
    for name in data:
        x = data[name]["x"]
        y = data[name]["y"]
        axs[0].plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()
