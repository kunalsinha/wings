import os
import glob
from PIL import Image
import numpy as np
from wings.utils.data import Dataset

class MNIST(Dataset):
    """
    Loads the MNIST dataset
    """

    def __init__(self, root, train=True, digits=None):
        """
        Read the MNIST images from the given path.

        Args:
            root (str): path to the MNIST dataset.
            train (bool): switch to load train or test data.
            digits (list): list of digits to be read. None implies all.
        """
        super().__init__()
        self.root = root
        self.train = train
        self.digits = digits
        # set path to train/test data
        self.path = self._build_path()
        return self._load_data()

    def _build_path(self):
        if self.train:
            return os.path.join(self.root, "training")
        else:
            return os.path.join(self.root, "testing")

    def _load_data(self):
        digits_features = []
        digits_labels = []
        if not self.digits:
            self.digits = list(range(10))
        print(self.digits)
        for digit in self.digits:
            digit_path = os.path.join(self.path, str(digit))
            mini_data, mini_labels = self._read_digits(digit, digit_path)
            digits_features += mini_data
            digits_labels += mini_labels
        self.data = np.array(digits_features)
        self.labels = np.array(digits_labels).reshape(-1, 1)

    def _read_digits(self, digit, path):
        # select .png files
        mini_data = []
        images = glob.glob(os.path.join(path, "*.png"))
        mini_labels = [digit] * len(images)
        for image in images:
            mini_data.append(self._read_image(image))
        return mini_data, mini_labels

    def _read_image(self, image):
        return np.array(Image.open(image)).ravel()

