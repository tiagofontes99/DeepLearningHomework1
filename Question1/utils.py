import random
import numpy as np
from matplotlib import pyplot as plt


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(data_path, bias):
    data = np.load(data_path, allow_pickle=bias)
    return data


def plot(param, param1, param2, filename):
    epoch , acc= param2["train"]
    epoch2 , valid = param2["valid"]
    plt.plot(epoch, acc , label="Train")
    plt.plot(epoch2, valid, label="Valid")
    plt.xlabel(param)
    plt.ylabel(param1)
    plt.title(filename)
    plt.grid(True)

    plt.show()
    plt.savefig(filename)
    return None
