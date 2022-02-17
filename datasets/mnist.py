import os
import glob
from PIL import Image
import numpy as np
from wings.utils.data import Dataset

class MNIST(Dataset):
    """
    Loads the MNIST dataset.
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
        self._load_data()

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


class FashionMNIST(Dataset):
    """
    Loads the FashionMNIST dataset.
    """

    def __init__(self, root, train=True):
        """
        Args:
            root: path to FashionMNIST dataset
            train: True for loading training data, False for test data
        """
        self.root = root
        self.train = train
        self.descriptions = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        self.data_file, self.label_file = self._get_files()
        self._load_dataset()

    def _get_files(self):
        """"
        Gets the appropriate files for train/test datasets.
        """
        train_data = 'train-images-idx3-ubyte'
        train_labels = "train-labels-idx1-ubyte"
        test_data = "t10k-images-idx3-ubyte"
        test_labels = "t10k-labels-idx1-ubyte"
        data_file, label_file = None, None
        if self.train:
            data_file = os.path.join(self.root, 'raw', train_data)
            label_file = os.path.join(self.root, 'raw', train_labels)
        else:
            data_file = os.path.join(self.root, 'raw', test_data)
            label_file = os.path.join(self.root, 'raw', test_labels)
        return (data_file, label_file)

    def _load_dataset(self):
        """
        Loads the images and labels.
        """
        self.data = self._read_data()
        self.labels = self._read_labels().reshape(-1, 1)

    def _read_data(self):
        """
        Reads the image data file.
        """
        with open(self.data_file, 'rb') as f:
            f.read(4)
            self.N = int.from_bytes(f.read(4), 'big')
            self.rows = int.from_bytes(f.read(4), 'big')
            self.cols = int.from_bytes(f.read(4), 'big')
            image_buffer = f.read(self.N * self.rows * self.cols)
        image_array = np.frombuffer(image_buffer, dtype=np.uint8)
        image_array = image_array.reshape(self.N, self.rows, self.cols)
        return image_array

    def _read_labels(self):
        """
        Reads the label file.
        """
        with open(self.label_file, 'rb') as f:
            f.read(4)
            N = int.from_bytes(f.read(4), 'big')
            label_buffer = f.read(N)
        label_array = np.frombuffer(label_buffer, dtype=np.uint8)
        return label_array

    def label_to_text(self):
        return self.descriptions

    def __repr__(self):
        if self.train:
            t = "training"
        else:
            t = "test"
        text = f"FashionMNIST {t} dataset\n"
        text += f"Number of images: {self.N}\n"
        text += f"Image shape: {self.rows} x {self.cols}\n"
        return text

